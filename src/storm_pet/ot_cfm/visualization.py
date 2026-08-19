#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualization utilities for latent spaces and decoded trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
from pathlib import Path

if sys.platform.startswith("win"):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from .utilities import normalize_bool_series, numeric_series, numpy_pca_2d, one_hot_subtype, silhouette_score_np
except ImportError:  # direct script execution from this folder
    from .utilities import normalize_bool_series, numeric_series, numpy_pca_2d, one_hot_subtype, silhouette_score_np


@dataclass
class EncodedLatentProjection:
    """Reusable 2D projection fitted on encoded latent z.

    Keeping the reducer and coordinates together lets the encoded-latent
    background, latent trajectories, and velocity arrows use the same UMAP/PCA
    coordinate system.
    """

    backend_name: str
    reducer: object
    coords: np.ndarray
    display_mask: np.ndarray
    fit_indices: np.ndarray


def visualize_latent(df: pd.DataFrame, z_np: np.ndarray, rec_mse: np.ndarray, args, out_dir: Path):
    # Prefer rows used for OT/FM, but fall back to rows with valid bin labels.
    mask = np.ones(len(df), dtype=bool)
    if args.use_col in df.columns and args.visualize_use_in_otfm_only:
        mask &= normalize_bool_series(df[args.use_col]).to_numpy()
    if args.bin_col in df.columns:
        mask &= pd.to_numeric(df[args.bin_col], errors="coerce").notna().to_numpy()
    if args.subtype_col in df.columns:
        mask &= pd.to_numeric(df[args.subtype_col], errors="coerce").notna().to_numpy()

    vis_df = df.loc[mask].copy()
    z_vis = z_np[mask]
    rec_vis = rec_mse[mask]

    if z_vis.shape[1] == 2:
        z2 = z_vis
        evr = np.array([np.nan, np.nan], dtype=np.float32)
    else:
        z2, evr = numpy_pca_2d(z_vis)

    vis_out = vis_df[["_source_row"]].copy()
    for k in range(z_vis.shape[1]):
        vis_out[f"z{k}"] = z_vis[:, k]
    vis_out["latent_vis_1"] = z2[:, 0]
    vis_out["latent_vis_2"] = z2[:, 1]
    vis_out["reconstruction_mse_standardized"] = rec_vis
    for col in [args.subtype_col, args.bin_col, args.stage_col, args.use_col, "Research Group", "PTID", "SCANDATE"]:
        if col in vis_df.columns:
            vis_out[col] = vis_df[col].to_numpy()
    vis_out.to_csv(out_dir / "latent_coordinates.csv", index=False)

    subtype_labels = pd.to_numeric(vis_df[args.subtype_col], errors="coerce").to_numpy()
    bin_labels = pd.to_numeric(vis_df[args.bin_col], errors="coerce").to_numpy()
    stage_labels = pd.to_numeric(vis_df[args.stage_col], errors="coerce").to_numpy() if args.stage_col in vis_df.columns else bin_labels

    sil_sub = silhouette_score_np(z2, subtype_labels, seed=args.seed)
    sil_bin = silhouette_score_np(z2, bin_labels, seed=args.seed)
    sil_stage = silhouette_score_np(z2, stage_labels, seed=args.seed)
    sep = pd.DataFrame(
        [
            {
                "latent_for_score": "2D visualization coordinates",
                "n_points": int(len(z2)),
                "pca_explained_var_1": float(evr[0]) if len(evr) > 0 else np.nan,
                "pca_explained_var_2": float(evr[1]) if len(evr) > 1 else np.nan,
                "silhouette_subtype": sil_sub,
                "silhouette_adaptive_bin": sil_bin,
                "silhouette_sustain_stage": sil_stage,
            }
        ]
    )
    sep.to_csv(out_dir / "latent_separation_summary.csv", index=False)

    def scatter_by_label(ax, labels, title, cmap=None):
        labels = np.asarray(labels)
        uniq = [u for u in np.unique(labels[~pd.isna(labels)])]
        for u in uniq:
            m = labels == u
            ax.scatter(z2[m, 0], z2[m, 1], s=18, alpha=0.75, label=str(int(u)) if float(u).is_integer() else str(u))
        ax.set_title(title)
        ax.set_xlabel("latent visualization dim 1")
        ax.set_ylabel("latent visualization dim 2")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, markerscale=1.3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter_by_label(axes[0], subtype_labels, f"Latent space by subtype; silhouette={sil_sub:.3f}")
    scatter_by_label(axes[1], bin_labels, f"Latent space by adaptive bin; silhouette={sil_bin:.3f}")
    fig.suptitle("Autoencoder latent representation")
    fig.tight_layout()
    fig.savefig(out_dir / "latent_space_subtype_and_bin.png", dpi=260)
    plt.close(fig)

    # Faceted plot: one panel per subtype, color by bin.
    subtype_unique = sorted(np.unique(subtype_labels[~pd.isna(subtype_labels)]).astype(int).tolist())
    n_cols = len(subtype_unique)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
    for ax, subtype in zip(axes.ravel(), subtype_unique):
        m_sub = subtype_labels == subtype
        for b in sorted(np.unique(bin_labels[m_sub & ~pd.isna(bin_labels)]).astype(int).tolist()):
            m = m_sub & (bin_labels == b)
            ax.scatter(z2[m, 0], z2[m, 1], s=18, alpha=0.75, label=f"bin {b}")
        ax.set_title(f"Subtype {subtype}")
        ax.set_xlabel("latent visualization dim 1")
        ax.set_ylabel("latent visualization dim 2")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Autoencoder latent space: adaptive bins within each subtype")
    fig.tight_layout()
    fig.savefig(out_dir / "latent_space_bins_within_subtype.png", dpi=300)
    plt.close(fig)

    # Reconstruction histogram.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(rec_mse, bins=40, alpha=0.85)
    ax.set_title("Autoencoder reconstruction error")
    ax.set_xlabel("per-subject MSE in standardized ROI space")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "autoencoder_reconstruction_mse_hist.png", dpi=300)
    plt.close(fig)

    return vis_out, sep


def _encoded_latent_display_mask(df: pd.DataFrame, args) -> np.ndarray:
    """Rows displayed in encoded-latent UMAP diagnostics.

    This mirrors the existing latent visualization filters so changes to the
    display flags or label columns affect all encoded-latent plots consistently.
    """
    mask = np.ones(len(df), dtype=bool)
    if args.use_col in df.columns and args.visualize_use_in_otfm_only:
        mask = mask & normalize_bool_series(df[args.use_col]).to_numpy(dtype=bool)
    if args.bin_col in df.columns:
        mask = mask & numeric_series(df[args.bin_col]).notna().to_numpy(dtype=bool)
    if args.subtype_col in df.columns:
        mask = mask & numeric_series(df[args.subtype_col]).notna().to_numpy(dtype=bool)
    return mask


def fit_encoded_latent_projection(df: pd.DataFrame, z_np: np.ndarray, args) -> EncodedLatentProjection | None:
    """Fit the reusable UMAP/PCA view for encoded latent z."""
    if z_np.shape[0] != len(df):
        raise ValueError(f"z_np has {z_np.shape[0]} rows, but df has {len(df)} rows.")
    if z_np.shape[1] < 2:
        print("WARNING: encoded latent UMAP needs at least 2 latent dimensions.")
        return None

    mask = _encoded_latent_display_mask(df, args)
    fit_idx = np.flatnonzero(mask)
    if len(fit_idx) < 3:
        print("WARNING: not enough encoded latent points for UMAP visualization.")
        return None

    # This limit affects only the encoded-latent UMAP fit. Keeping it separate
    # from --umap_max_fit_points makes it easy to tune this diagnostic plot
    # without changing the ROI-space trajectory overlay.
    max_fit_points = getattr(args, "encoded_umap_max_fit_points", 0)
    if max_fit_points is None:
        max_fit_points = 0
    rng = np.random.default_rng(args.seed)
    if max_fit_points > 0 and len(fit_idx) > max_fit_points:
        fit_idx = rng.choice(fit_idx, size=max_fit_points, replace=False)

    backend_name, reducer = _fit_projection_2d(z_np[fit_idx].astype(np.float32), args)
    coords = reducer.transform(z_np.astype(np.float32))
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[1] < 2:
        raise ValueError(f"Encoded latent projection returned {coords.shape[1]} dimensions; expected 2.")

    return EncodedLatentProjection(
        backend_name=backend_name,
        reducer=reducer,
        coords=coords,
        display_mask=mask,
        fit_indices=fit_idx,
    )


def visualize_encoded_latent_umap(
    df: pd.DataFrame,
    z_np: np.ndarray,
    rec_mse: np.ndarray,
    args,
    out_dir: Path,
    return_projection: bool = False,
):
    """Run a separate UMAP/PCA projection directly on the encoded latent vectors.

    The existing trajectory UMAP is fit in standardized 68-ROI SUVR space so that
    decoded ROI trajectories can be overlaid with real ROI observations. This
    function is intentionally different: it fits UMAP/PCA on the autoencoder
    encoded vectors z themselves. Adjust --umap_* arguments in Main.py to tune
    this view; use --encoded_umap_max_fit_points to subsample only this plot.
    """
    if not getattr(args, "plot_encoded_umap", True):
        empty = (pd.DataFrame(), pd.DataFrame())
        return (*empty, None) if return_projection else empty

    projection = fit_encoded_latent_projection(df=df, z_np=z_np, args=args)
    if projection is None:
        empty = (pd.DataFrame(), pd.DataFrame())
        return (*empty, None) if return_projection else empty

    backend_name = projection.backend_name
    coords = projection.coords
    mask = projection.display_mask
    fit_idx = projection.fit_indices

    coord_df = pd.DataFrame(
        {
            "_source_row": df["_source_row"].to_numpy() if "_source_row" in df.columns else np.arange(len(df)),
            "encoded_umap1": coords[:, 0],
            "encoded_umap2": coords[:, 1],
            "used_for_encoded_umap_display": mask,
            "reconstruction_mse_standardized": rec_mse,
        }
    )
    for k in range(z_np.shape[1]):
        coord_df[f"z{k}"] = z_np[:, k]
    for col in [
        args.subtype_col,
        args.bin_col,
        args.stage_col,
        args.use_col,
        getattr(args, "dynamics_label_col", ""),
        "PTID",
        "SCANDATE",
        "Research Group",
    ]:
        if col and col in df.columns:
            coord_df[col] = df[col].to_numpy()
    coord_df.to_csv(out_dir / f"{backend_name}_encoded_latent_coordinates.csv", index=False)

    plot_df = df.loc[mask].copy()
    plot_coords = coords[mask]
    subtype_labels = numeric_series(plot_df[args.subtype_col]).to_numpy() if args.subtype_col in plot_df.columns else np.full(len(plot_df), np.nan)
    bin_values = numeric_series(plot_df[args.bin_col]).to_numpy() if args.bin_col in plot_df.columns else np.full(len(plot_df), np.nan)
    stage_labels = numeric_series(plot_df[args.stage_col]).to_numpy() if args.stage_col in plot_df.columns else bin_values

    sil_sub = silhouette_score_np(plot_coords, subtype_labels, seed=args.seed)
    sil_bin = silhouette_score_np(plot_coords, bin_values, seed=args.seed)
    sil_stage = silhouette_score_np(plot_coords, stage_labels, seed=args.seed)
    summary = pd.DataFrame(
        [
            {
                "projection_source": "encoded latent z",
                "projection_backend": backend_name,
                "n_points": int(len(plot_coords)),
                "n_fit_points": int(len(fit_idx)),
                "silhouette_subtype": sil_sub,
                "silhouette_adaptive_bin": sil_bin,
                "silhouette_sustain_stage": sil_stage,
            }
        ]
    )
    summary.to_csv(out_dir / f"{backend_name}_encoded_latent_separation_summary.csv", index=False)

    bin_label_lookup = _bin_label_map(plot_df, args)

    def scatter_by_label(ax, labels, title, label_prefix=""):
        labels = np.asarray(labels)
        valid = ~pd.isna(labels)
        for lab in sorted(np.unique(labels[valid]).astype(int).tolist()):
            m = valid & (labels.astype(float) == float(lab))
            display_label = bin_label_lookup.get(lab, f"{label_prefix}{lab}") if label_prefix == "bin " else f"{label_prefix}{lab}"
            ax.scatter(plot_coords[m, 0], plot_coords[m, 1], s=18, alpha=0.72, label=display_label)
        ax.set_title(title)
        ax.set_xlabel(f"{backend_name.upper()}1 of encoded z")
        ax.set_ylabel(f"{backend_name.upper()}2 of encoded z")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, markerscale=1.3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter_by_label(axes[0], subtype_labels, f"Encoded latent {backend_name.upper()} by subtype; silhouette={sil_sub:.3f}", "subtype ")
    scatter_by_label(axes[1], bin_values, f"Encoded latent {backend_name.upper()} by dynamics bin; silhouette={sil_bin:.3f}", "bin ")
    fig.suptitle("Direct UMAP/PCA projection of autoencoder encoded latent z")
    fig.tight_layout()
    fig.savefig(out_dir / f"{backend_name}_encoded_latent_by_subtype_and_bin.png", dpi=260)
    plt.close(fig)

    # Faceted diagnostic: each subtype gets its own panel, with color showing
    # dynamics bins. This is the easiest plot to modify if you want one subtype
    # per figure or a different grouping variable later.
    subtype_unique = sorted(np.unique(subtype_labels[~pd.isna(subtype_labels)]).astype(int).tolist())
    n_cols = max(len(subtype_unique), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
    for ax, subtype in zip(axes.ravel(), subtype_unique):
        m_sub = subtype_labels == subtype
        for b in sorted(np.unique(bin_values[m_sub & ~pd.isna(bin_values)]).astype(int).tolist()):
            m = m_sub & (bin_values == b)
            ax.scatter(plot_coords[m, 0], plot_coords[m, 1], s=18, alpha=0.75, label=bin_label_lookup.get(b, f"bin {b}"))
        ax.set_title(f"Subtype {subtype}")
        ax.set_xlabel(f"{backend_name.upper()}1 of encoded z")
        ax.set_ylabel(f"{backend_name.upper()}2 of encoded z")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    if not subtype_unique:
        axes.ravel()[0].axis("off")
    fig.suptitle("Encoded latent projection: dynamics bins within each subtype")
    fig.tight_layout()
    fig.savefig(out_dir / f"{backend_name}_encoded_latent_bins_within_subtype.png", dpi=260)
    plt.close(fig)

    if return_projection:
        return coord_df, summary, projection
    return coord_df, summary


def _plot_encoded_latent_background(ax, coords: np.ndarray, df: pd.DataFrame, mask: np.ndarray, args, title: str) -> dict[int, str]:
    """Draw observed encoded-latent points as open circles colored by dynamics bin."""
    plot_df = df.loc[mask].copy()
    plot_coords = coords[mask]
    bin_values = numeric_series(plot_df[args.bin_col]).to_numpy() if args.bin_col in plot_df.columns else np.full(len(plot_df), np.nan)
    bin_labels = _bin_label_map(plot_df, args)

    valid = ~pd.isna(bin_values)
    for b in sorted(np.unique(bin_values[valid]).astype(int).tolist()):
        m = valid & (bin_values == b)
        ax.scatter(
            plot_coords[m, 0],
            plot_coords[m, 1],
            s=18,
            facecolors="none",
            edgecolors=plt.cm.tab10(b % 10),
            linewidths=0.8,
            alpha=0.28,
            label=f"observed {bin_labels.get(b, f'bin {b}')}",
        )
    ax.set_title(title)
    ax.set_xlabel("Encoded UMAP/PCA 1")
    ax.set_ylabel("Encoded UMAP/PCA 2")
    ax.grid(alpha=0.22)
    return bin_labels


def _clip_velocity(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return v
    norm = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(float(max_norm) / norm, max=1.0)
    return v * scale


def _parse_velocity_steps(args) -> list[float]:
    raw_steps = getattr(args, "encoded_velocity_steps", "")
    if raw_steps is None:
        raw_steps = ""
    if isinstance(raw_steps, str) and raw_steps.strip():
        steps = [float(x.strip()) for x in raw_steps.split(",") if x.strip()]
    elif isinstance(raw_steps, (list, tuple, np.ndarray)):
        steps = [float(x) for x in raw_steps]
    else:
        steps = [float(getattr(args, "encoded_velocity_step", 0.05))]
    steps = [s for s in steps if np.isfinite(s) and s > 0]
    return steps or [0.05]


def _step_label(step: float) -> str:
    return f"{step:g}".replace("-", "m").replace(".", "p")


def _time_normalization(values: np.ndarray, all_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite_all = np.asarray(all_values, dtype=np.float32)
    finite_all = finite_all[np.isfinite(finite_all)]
    if len(finite_all) == 0:
        return np.zeros_like(values, dtype=np.float32)
    min_v = float(np.nanmin(finite_all))
    max_v = float(np.nanmax(finite_all))
    if max_v > min_v:
        return ((values - min_v) / max(max_v - min_v, 1e-8)).astype(np.float32)
    return np.zeros_like(values, dtype=np.float32)


def _project_velocity_delta(
    projection: EncodedLatentProjection,
    z_start_np: np.ndarray,
    v_np: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project latent velocity to 2D.

    UMAP uses finite difference phi(z + h v) - phi(z). PCA uses the linear
    components directly, and also returns finite-difference deltas for a
    consistency diagnostic.
    """
    z_start_np = np.asarray(z_start_np, dtype=np.float32)
    v_np = np.asarray(v_np, dtype=np.float32)
    z_end_np = z_start_np + float(step) * v_np
    start_coords = np.asarray(projection.reducer.transform(z_start_np), dtype=np.float32)
    end_coords = np.asarray(projection.reducer.transform(z_end_np), dtype=np.float32)
    finite_delta = end_coords - start_coords

    components = getattr(projection.reducer, "components_", None)
    if projection.backend_name == "pca" and components is not None:
        linear_delta = float(step) * (v_np @ np.asarray(components[:2], dtype=np.float32).T)
        delta = np.asarray(linear_delta, dtype=np.float32)
    else:
        delta = np.asarray(finite_delta, dtype=np.float32)
    return start_coords, delta, finite_delta, z_end_np


def _clip_or_normalize_arrows(delta: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, float]:
    delta = np.asarray(delta, dtype=np.float32)
    raw_norm = np.linalg.norm(delta, axis=1)
    adjusted = delta.copy()
    clip_quantile = getattr(args, "encoded_velocity_arrow_clip_quantile", 0.95)
    clip_value = np.nan
    if clip_quantile is not None and 0 < float(clip_quantile) < 1 and len(raw_norm):
        clip_value = float(np.nanquantile(raw_norm, float(clip_quantile)))
        if np.isfinite(clip_value) and clip_value > 0:
            scale = np.minimum(1.0, clip_value / np.maximum(raw_norm, 1e-8))
            adjusted = adjusted * scale[:, None]

    if getattr(args, "encoded_velocity_arrow_normalize", False) and len(raw_norm):
        fixed_length = float(getattr(args, "encoded_velocity_arrow_length", 0.06))
        adjusted_norm = np.linalg.norm(adjusted, axis=1)
        adjusted = adjusted / np.maximum(adjusted_norm[:, None], 1e-8) * fixed_length

    return adjusted, raw_norm, clip_value


def _velocity_summary(rows: list[dict], step: float, backend_name: str, cfm_input_space: str, projection_input_space: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            [
                {
                    "projection_backend": backend_name,
                    "cfm_input_space": cfm_input_space,
                    "projection_input_space": projection_input_space,
                    "encoded_velocity_step": step,
                    "n_vectors": 0,
                }
            ]
        )
    df_rows = pd.DataFrame(rows)

    def stats(prefix: str, values) -> dict:
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            return {f"mean_{prefix}": np.nan, f"median_{prefix}": np.nan, f"p95_{prefix}": np.nan}
        return {
            f"mean_{prefix}": float(np.nanmean(arr)),
            f"median_{prefix}": float(np.nanmedian(arr)),
            f"p95_{prefix}": float(np.nanpercentile(arr, 95)),
        }

    out = {
        "projection_backend": backend_name,
        "cfm_input_space": cfm_input_space,
        "projection_input_space": projection_input_space,
        "encoded_velocity_step": step,
        "n_vectors": int(len(df_rows)),
    }
    out.update(stats("v_norm", df_rows["v_norm"]))
    out.update(stats("step_norm", df_rows["step_norm"]))
    out.update(stats("umap_delta_norm", df_rows["umap_delta_norm"]))
    return pd.DataFrame([out])


def plot_encoded_latent_dynamics_on_umap(
    latent_traj_by_subtype: dict[int, np.ndarray],
    cfm_model,
    df: pd.DataFrame,
    z_np: np.ndarray,
    prob_cols: list[str],
    args,
    out_dir: Path,
    device: torch.device,
    t_eval: np.ndarray,
    projection: EncodedLatentProjection | None = None,
    root_anchor_z: np.ndarray | None = None,
    sample_traj_by_subtype: dict[int, np.ndarray] | None = None,
    trajectory_initial_source: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overlay learned latent trajectories and velocity arrows on encoded UMAP/PCA.

    CFM was trained on raw autoencoder latent z and raw bin-time coordinates
    t = bin_index + local_t. The vector-field diagnostic therefore evaluates
    v_theta in that same input space and saves a separate normalized time column
    only for plotting/diagnostics.
    """
    if not getattr(args, "plot_encoded_dynamics_umap", True):
        return pd.DataFrame(), pd.DataFrame()
    if not getattr(args, "plot_encoded_umap", True):
        return pd.DataFrame(), pd.DataFrame()
    if not latent_traj_by_subtype:
        return pd.DataFrame(), pd.DataFrame()

    if projection is None:
        projection = fit_encoded_latent_projection(df=df, z_np=z_np, args=args)
    if projection is None:
        return pd.DataFrame(), pd.DataFrame()

    backend_name = projection.backend_name
    coords = projection.coords
    mask = projection.display_mask
    rng = np.random.default_rng(args.seed)
    time_values = np.asarray(t_eval, dtype=np.float32)
    all_bin_values = numeric_series(df[args.bin_col]).to_numpy(dtype=np.float32) if args.bin_col in df.columns else time_values
    all_time_reference = np.concatenate([all_bin_values[np.isfinite(all_bin_values)], time_values[np.isfinite(time_values)]])
    cfm_input_space = "raw_encoded_z"
    projection_input_space = "raw_encoded_z"
    subtype_colors = plt.cm.tab10(np.linspace(0, 1, max(len(latent_traj_by_subtype), 1)))
    initial_source = trajectory_initial_source or ("shared_root" if getattr(args, "shared_root_stage0", False) else "subtype_specific")

    trajectory_rows = []
    trajectory_velocity_source_rows = []
    for color_i, subtype in enumerate(sorted(latent_traj_by_subtype)):
        z_traj = np.asarray(latent_traj_by_subtype[subtype], dtype=np.float32)
        if z_traj.ndim != 3:
            raise ValueError(f"Expected latent trajectory for subtype {subtype} to have shape T x N x D.")

        n_time, n_samples, latent_dim = z_traj.shape
        traj_coords = projection.reducer.transform(z_traj.reshape(n_time * n_samples, latent_dim)).reshape(n_time, n_samples, 2)
        traj_coords = np.asarray(traj_coords, dtype=np.float32)
        np.save(out_dir / f"{backend_name}_encoded_latent_traj_coords_subtype_{subtype}.npy", traj_coords)

        projected_mean_path = traj_coords.mean(axis=1)
        latent_mean_path = np.asarray(projection.reducer.transform(z_traj.mean(axis=1)), dtype=np.float32)
        time_for_rows = time_values[:n_time] if len(time_values) >= n_time else np.arange(n_time, dtype=np.float32)
        time_norm_for_rows = _time_normalization(time_for_rows, all_time_reference)
        for ti in range(n_time):
            trajectory_rows.append(
                {
                    "sustain_subtype": int(subtype),
                    "branch_subtype": int(subtype),
                    "initial_source": initial_source,
                    "time_index": int(ti),
                    "cfm_time": float(time_for_rows[ti]),
                    "adaptive_bin_raw": float(time_for_rows[ti]),
                    "adaptive_time_normalized": float(time_norm_for_rows[ti]),
                    "encoded_umap1_projected_mean": float(projected_mean_path[ti, 0]),
                    "encoded_umap2_projected_mean": float(projected_mean_path[ti, 1]),
                    "encoded_umap1_latent_mean": float(latent_mean_path[ti, 0]),
                    "encoded_umap2_latent_mean": float(latent_mean_path[ti, 1]),
                }
            )

        sample_count = min(getattr(args, "n_encoded_umap_trajectory_samples", 12), n_samples)
        sample_idx = rng.choice(n_samples, size=sample_count, replace=False) if sample_count > 0 else []
        flat_candidates = []
        for ti in range(n_time):
            for sample_id in sample_idx:
                flat_candidates.append((ti, int(sample_id)))
        if flat_candidates:
            max_traj_arrows = min(getattr(args, "n_encoded_velocity_arrows", 80), len(flat_candidates))
            picked = rng.choice(len(flat_candidates), size=max_traj_arrows, replace=False)
            for pick in picked:
                ti, sample_id = flat_candidates[int(pick)]
                trajectory_velocity_source_rows.append(
                    {
                        "velocity_source": "trajectory",
                        "subtype": int(subtype),
                        "branch_subtype": int(subtype),
                        "initial_source": initial_source,
                        "time_index": int(ti),
                        "trajectory_id": int(sample_id),
                        "z": z_traj[ti, sample_id],
                        "cfm_time": float(time_for_rows[ti]),
                        "adaptive_bin_raw": float(time_for_rows[ti]),
                        "adaptive_time_normalized": float(_time_normalization(np.array([time_for_rows[ti]], dtype=np.float32), all_time_reference)[0]),
                    }
                )

    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_df.to_csv(out_dir / f"{backend_name}_encoded_latent_trajectory_mean_path_by_subtype.csv", index=False)

    observed_sources = []
    valid_velocity = mask.copy()
    if args.subtype_col in df.columns:
        valid_velocity = valid_velocity & numeric_series(df[args.subtype_col]).notna().to_numpy(dtype=bool)
    if args.bin_col in df.columns:
        valid_velocity = valid_velocity & numeric_series(df[args.bin_col]).notna().to_numpy(dtype=bool)
    subtype_all = numeric_series(df[args.subtype_col]).to_numpy(dtype=np.float32) if args.subtype_col in df.columns else np.full(len(df), np.nan)
    valid_velocity = valid_velocity & np.isfinite(subtype_all)
    valid_velocity = valid_velocity & (subtype_all >= 0) & (subtype_all < len(prob_cols))
    candidate_idx = np.flatnonzero(valid_velocity)
    n_arrows = min(getattr(args, "n_encoded_velocity_arrows", 80), len(candidate_idx))
    if n_arrows > 0:
        chosen_idx = rng.choice(candidate_idx, size=n_arrows, replace=False)
        chosen_rows = df.iloc[chosen_idx]
        bin_raw = numeric_series(chosen_rows[args.bin_col]).to_numpy(dtype=np.float32)
        bin_norm = _time_normalization(bin_raw, all_time_reference)
        subtype_values = numeric_series(chosen_rows[args.subtype_col]).astype(int).to_numpy()
        for local_i, source_idx in enumerate(chosen_idx):
            observed_sources.append(
                {
                    "velocity_source": "observed",
                    "_source_row": int(df["_source_row"].iloc[source_idx]) if "_source_row" in df.columns else int(source_idx),
                    "sustain_subtype": int(subtype_values[local_i]),
                    "z": z_np[source_idx].astype(np.float32),
                    "cfm_time": float(bin_raw[local_i]),
                    "adaptive_bin_raw": float(bin_raw[local_i]),
                    "adaptive_time_normalized": float(bin_norm[local_i]),
                }
            )

    velocity_steps = _parse_velocity_steps(args)
    selected_mean_mode = getattr(args, "mean_path_mode", "projected_mean")
    if selected_mean_mode not in {"projected_mean", "latent_mean"}:
        selected_mean_mode = "projected_mean"

    all_velocity_rows = []
    for step_index, step in enumerate(velocity_steps):
        step_label = _step_label(step)
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
        _plot_encoded_latent_background(axes[0], coords, df, mask, args, "Observed-point velocity field")
        _plot_encoded_latent_background(axes[1], coords, df, mask, args, "Trajectory-point velocity field")
        _plot_encoded_latent_background(axes[2], coords, df, mask, args, f"Learned latent trajectories ({selected_mean_mode})")

        velocity_rows = []

        def add_velocity_vectors(source_rows: list[dict], ax) -> None:
            if not source_rows:
                return
            z_start_np = np.vstack([row["z"] for row in source_rows]).astype(np.float32)
            subtype_np = np.array([row["sustain_subtype"] if "sustain_subtype" in row else row["subtype"] for row in source_rows], dtype=int)
            cfm_time_np = np.array([row["cfm_time"] for row in source_rows], dtype=np.float32)
            v_np = np.zeros_like(z_start_np, dtype=np.float32)
            cfm_model.eval()
            with torch.no_grad():
                for subtype in sorted(np.unique(subtype_np).astype(int).tolist()):
                    local_indices = np.flatnonzero(subtype_np == subtype)
                    z_local = torch.from_numpy(z_start_np[local_indices]).to(device)
                    t_local = torch.from_numpy(cfm_time_np[local_indices]).to(device)
                    cond = one_hot_subtype(subtype, len(prob_cols), len(local_indices), device)
                    v_local = cfm_model(z_local, t_local, cond)
                    v_local = _clip_velocity(v_local, getattr(args, "latent_velocity_clip", 0.0))
                    v_np[local_indices] = v_local.detach().cpu().numpy()

            start_coords, delta_raw, finite_delta, _ = _project_velocity_delta(projection, z_start_np, v_np, step)
            delta_plot, raw_delta_norm, clip_value = _clip_or_normalize_arrows(delta_raw, args)
            ax.quiver(
                start_coords[:, 0],
                start_coords[:, 1],
                delta_plot[:, 0],
                delta_plot[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="black",
                width=0.004,
                alpha=0.82,
            )
            ax.scatter(start_coords[:, 0], start_coords[:, 1], c=cfm_time_np, cmap="plasma", s=32, zorder=4)

            for row_i, row in enumerate(source_rows):
                v_norm = float(np.linalg.norm(v_np[row_i]))
                step_norm = float(step * v_norm)
                umap_delta_norm = float(raw_delta_norm[row_i])
                finite_delta_norm = float(np.linalg.norm(finite_delta[row_i]))
                velocity_rows.append(
                    {
                        "velocity_source": row["velocity_source"],
                        "_source_row": row.get("_source_row", -1),
                        "sustain_subtype": int(subtype_np[row_i]),
                        "branch_subtype": row.get("branch_subtype", int(subtype_np[row_i])),
                        "initial_source": row.get("initial_source", ""),
                        "trajectory_id": row.get("trajectory_id", -1),
                        "time_index": row.get("time_index", -1),
                        "cfm_time": float(row["cfm_time"]),
                        "adaptive_bin_raw": float(row["adaptive_bin_raw"]),
                        "adaptive_time_normalized": float(row["adaptive_time_normalized"]),
                        "encoded_velocity_step": float(step),
                        "cfm_input_space": cfm_input_space,
                        "projection_input_space": projection_input_space,
                        "projection_backend": backend_name,
                        "velocity_projection_method": "pca_linear" if backend_name == "pca" and getattr(projection.reducer, "components_", None) is not None else "finite_difference",
                        "encoded_umap1": float(start_coords[row_i, 0]),
                        "encoded_umap2": float(start_coords[row_i, 1]),
                        "velocity_umap_dx": float(delta_raw[row_i, 0]),
                        "velocity_umap_dy": float(delta_raw[row_i, 1]),
                        "velocity_umap_dx_plotted": float(delta_plot[row_i, 0]),
                        "velocity_umap_dy_plotted": float(delta_plot[row_i, 1]),
                        "finite_difference_dx": float(finite_delta[row_i, 0]),
                        "finite_difference_dy": float(finite_delta[row_i, 1]),
                        "z_norm": float(np.linalg.norm(z_start_np[row_i])),
                        "v_norm": v_norm,
                        "step_norm": step_norm,
                        "umap_delta_norm": umap_delta_norm,
                        "finite_difference_delta_norm": finite_delta_norm,
                        "arrow_delta_norm_plotted": float(np.linalg.norm(delta_plot[row_i])),
                        "arrow_clip_value": clip_value,
                        "arrow_normalized": bool(getattr(args, "encoded_velocity_arrow_normalize", False)),
                    }
                )

        add_velocity_vectors(observed_sources, axes[0])
        add_velocity_vectors(trajectory_velocity_source_rows, axes[1])

        shared_root_points = []
        sample_path_label_added = False
        for color_i, subtype in enumerate(sorted(latent_traj_by_subtype)):
            z_traj = np.asarray(latent_traj_by_subtype[subtype], dtype=np.float32)
            n_time, n_samples, latent_dim = z_traj.shape
            traj_coords = np.asarray(
                projection.reducer.transform(z_traj.reshape(n_time * n_samples, latent_dim)).reshape(n_time, n_samples, 2),
                dtype=np.float32,
            )
            if sample_traj_by_subtype is None:
                sample_count = min(getattr(args, "n_encoded_umap_trajectory_samples", 12), n_samples)
                sample_idx = rng.choice(n_samples, size=sample_count, replace=False) if sample_count > 0 else []
                for j in sample_idx:
                    axes[2].plot(traj_coords[:, j, 0], traj_coords[:, j, 1], color="0.35", alpha=0.28, linewidth=1.0)
            else:
                sample_z_traj = sample_traj_by_subtype.get(int(subtype))
                if sample_z_traj is not None:
                    sample_z_traj = np.asarray(sample_z_traj, dtype=np.float32)
                    if sample_z_traj.ndim != 3:
                        raise ValueError(f"Expected sample trajectories for subtype {subtype} to have shape T x N x D.")
                    sample_time, sample_count, sample_dim = sample_z_traj.shape
                    sample_coords = np.asarray(
                        projection.reducer.transform(sample_z_traj.reshape(sample_time * sample_count, sample_dim)).reshape(sample_time, sample_count, 2),
                        dtype=np.float32,
                    )
                    for j in range(sample_count):
                        label = "root sample paths" if not sample_path_label_added else None
                        axes[2].plot(
                            sample_coords[:, j, 0],
                            sample_coords[:, j, 1],
                            color="0.55",
                            alpha=0.12,
                            linewidth=0.9,
                            label=label,
                            zorder=2,
                        )
                        sample_path_label_added = True

            rows_sub = trajectory_df[trajectory_df["sustain_subtype"] == int(subtype)]
            if selected_mean_mode == "latent_mean":
                mean_x = rows_sub["encoded_umap1_latent_mean"].to_numpy(dtype=float)
                mean_y = rows_sub["encoded_umap2_latent_mean"].to_numpy(dtype=float)
            else:
                mean_x = rows_sub["encoded_umap1_projected_mean"].to_numpy(dtype=float)
                mean_y = rows_sub["encoded_umap2_projected_mean"].to_numpy(dtype=float)
            line_color = subtype_colors[color_i]
            axes[2].plot(mean_x, mean_y, color=line_color, linewidth=3.4, marker="o", markersize=4, label=f"mean path subtype {subtype}")
            if len(mean_x):
                shared_root_points.append((float(mean_x[0]), float(mean_y[0])))
            axes[2].scatter(
                mean_x,
                mean_y,
                c=rows_sub["cfm_time"].to_numpy(dtype=float),
                cmap="plasma",
                s=42,
                edgecolors="black",
                linewidths=0.35,
                zorder=5,
            )

        if root_anchor_z is not None:
            root_z = np.asarray(root_anchor_z, dtype=np.float32).reshape(1, -1)
            root_xy = np.asarray(projection.reducer.transform(root_z), dtype=np.float32).reshape(-1, 2)[0]
        elif getattr(args, "shared_root_stage0", False) and shared_root_points:
            root_xy = np.asarray(shared_root_points, dtype=np.float32).mean(axis=0)
        else:
            root_xy = None

        if root_xy is not None:
            axes[2].scatter(
                [root_xy[0]],
                [root_xy[1]],
                color="black",
                s=95,
                marker="s",
                label="shared root / stage0",
                zorder=7,
            )

        velocity_df_step = pd.DataFrame(velocity_rows)
        velocity_df_step.to_csv(out_dir / f"{backend_name}_encoded_latent_velocity_field_vectors_step_{step_label}.csv", index=False)
        _velocity_summary(velocity_rows, step, backend_name, cfm_input_space, projection_input_space).to_csv(
            out_dir / f"{backend_name}_encoded_latent_velocity_field_summary_step_{step_label}.csv",
            index=False,
        )
        all_velocity_rows.extend(velocity_rows)

        for ax in axes:
            ax.legend(frameon=False, fontsize=8, loc="best")
        fig.suptitle(f"Learned OT-CFM dynamics in encoded latent {backend_name.upper()} space, h={step:g}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{backend_name}_encoded_latent_velocity_and_trajectory_step_{step_label}.png", dpi=280)
        if step_index == 0:
            # Backward-compatible filename for notebooks/scripts that expect one
            # combined encoded dynamics figure.
            fig.savefig(out_dir / f"{backend_name}_encoded_latent_velocity_and_trajectory.png", dpi=280)
        plt.close(fig)

    velocity_df = pd.DataFrame(all_velocity_rows)
    velocity_df.to_csv(out_dir / f"{backend_name}_encoded_latent_velocity_field_vectors.csv", index=False)
    return trajectory_df, velocity_df


def _decoded_roi_grid_shape(n_plots: int, requested_rows: int = 0, requested_cols: int = 9) -> tuple[int, int]:
    n_plots = max(0, int(n_plots))
    if n_plots == 0:
        return 1, max(1, int(requested_cols) if requested_cols else 1)

    n_cols = int(requested_cols) if requested_cols and requested_cols > 0 else int(math.ceil(math.sqrt(n_plots)))
    n_cols = max(1, n_cols)
    n_rows = int(requested_rows) if requested_rows and requested_rows > 0 else int(math.ceil(n_plots / n_cols))
    n_rows = max(1, n_rows)
    if n_rows * n_cols < n_plots:
        n_rows = int(math.ceil(n_plots / n_cols))
    return n_rows, n_cols


def _decoded_roi_summary_dataframe(x_traj: np.ndarray, roi_cols: list[str], t_eval: np.ndarray) -> pd.DataFrame:
    x = np.asarray(x_traj, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("x_traj must have shape T x N x ROI.")
    if x.shape[2] != len(roi_cols):
        raise ValueError(f"x_traj ROI dimension {x.shape[2]} does not match len(roi_cols)={len(roi_cols)}.")

    t = np.asarray(t_eval, dtype=float)
    if len(t) != x.shape[0]:
        raise ValueError(f"t_eval length {len(t)} does not match trajectory time length {x.shape[0]}.")

    mean = np.mean(x, axis=1)
    variance = np.var(x, axis=1)
    std = np.sqrt(variance)
    sem = std / math.sqrt(max(x.shape[1], 1))

    rows = []
    for roi_idx, roi_name in enumerate(roi_cols):
        for ti, t_value in enumerate(t):
            rows.append(
                {
                    "roi_index": int(roi_idx),
                    "roi_name": roi_name,
                    "time_index": int(ti),
                    "adaptive_time": float(t_value),
                    "decoded_mean": float(mean[ti, roi_idx]),
                    "decoded_variance": float(variance[ti, roi_idx]),
                    "decoded_std": float(std[ti, roi_idx]),
                    "decoded_sem": float(sem[ti, roi_idx]),
                    "n_trajectories": int(x.shape[1]),
                }
            )
    return pd.DataFrame(rows)


def _selected_decoded_roi_indices(roi_cols: list[str], args, subtype: int) -> np.ndarray:
    if bool(getattr(args, "plot_all_decoded_rois", True)):
        return np.arange(len(roi_cols), dtype=int)
    rng = np.random.default_rng(args.seed + subtype)
    return np.sort(
        rng.choice(
            len(roi_cols),
            size=min(args.n_plot_rois, len(roi_cols)),
            replace=False,
        )
    ).astype(int)


def plot_decoded_trajectories(subtype: int, pools, roi_cols, x_traj: np.ndarray, df: pd.DataFrame, roi_mean: np.ndarray, roi_std: np.ndarray, args, out_dir: Path):
    selected_indices = _selected_decoded_roi_indices(roi_cols, args, subtype)

    # real boxplots from hard subtype/bin assignments
    work = df.copy()
    if args.use_col in work.columns and args.require_use_in_otfm:
        work = work[normalize_bool_series(work[args.use_col])].copy()
    bin_work = pd.to_numeric(work[args.bin_col], errors="coerce")
    subtype_work = pd.to_numeric(work[args.subtype_col], errors="coerce")
    if getattr(args, "shared_root_stage0", False):
        root_bin = getattr(args, "root_bin_value", 0)
        keep = ((bin_work == root_bin) | (subtype_work == subtype)) & bin_work.notna()
    else:
        keep = (subtype_work == subtype) & bin_work.notna()
    work = work[keep].copy()
    bin_ids = sorted(pd.to_numeric(work[args.bin_col], errors="coerce").dropna().astype(int).unique().tolist())

    t_eval = np.linspace(min(bin_ids), max(bin_ids), x_traj.shape[0]) if len(bin_ids) else np.arange(x_traj.shape[0])
    summary_df = _decoded_roi_summary_dataframe(x_traj=x_traj, roi_cols=roi_cols, t_eval=t_eval)
    summary_df.to_csv(out_dir / f"decoded_roi_mean_variance_subtype_{subtype}.csv", index=False)

    pd.DataFrame(
        {
            "selected_index": selected_indices,
            "roi_name": [roi_cols[i] for i in selected_indices],
        }
    ).to_csv(out_dir / f"selected_rois_for_decoded_plot_subtype_{subtype}.csv", index=False)

    n_plots = len(selected_indices)
    n_rows, n_cols = _decoded_roi_grid_shape(
        n_plots=n_plots,
        requested_rows=getattr(args, "decoded_plot_n_rows", 0),
        requested_cols=getattr(args, "decoded_plot_n_cols", 9),
    )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(14, 2.55 * n_cols), max(7, 2.25 * n_rows)),
        sharex=False,
        sharey=False,
    )
    axes = np.asarray(axes).reshape(-1)
    uncertainty_mode = getattr(args, "decoded_uncertainty_mode", "std")

    for ax, roi_idx in zip(axes, selected_indices):
        box_data = []
        positions = []
        for b in bin_ids:
            bin_numeric = pd.to_numeric(work[args.bin_col], errors="coerce").astype(int)
            if getattr(args, "shared_root_stage0", False) and b == getattr(args, "root_bin_value", 0):
                m = bin_numeric == b
            else:
                m = (pd.to_numeric(work[args.subtype_col], errors="coerce") == subtype) & (bin_numeric == b)
            vals = pd.to_numeric(work.loc[m, roi_cols[roi_idx]], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                box_data.append(vals)
                positions.append(b)

        if box_data:
            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.45,
                showfliers=False,
                patch_artist=True,
            )
            for patch in bp.get("boxes", []):
                patch.set_facecolor("0.88")
                patch.set_alpha(0.5)
            for item in bp.get("medians", []):
                item.set_color("0.35")

        roi_summary = summary_df[summary_df["roi_index"] == int(roi_idx)]
        mean_values = roi_summary["decoded_mean"].to_numpy(dtype=float)
        if uncertainty_mode == "variance":
            spread_values = roi_summary["decoded_variance"].to_numpy(dtype=float)
            band_label = "decoded mean +/- variance"
        elif uncertainty_mode == "sem":
            spread_values = roi_summary["decoded_sem"].to_numpy(dtype=float)
            band_label = "decoded mean +/- SEM"
        elif uncertainty_mode == "none":
            spread_values = None
            band_label = ""
        else:
            spread_values = roi_summary["decoded_std"].to_numpy(dtype=float)
            band_label = "decoded mean +/- SD"

        if spread_values is not None:
            lower = mean_values - spread_values
            upper = mean_values + spread_values
            ax.fill_between(
                t_eval,
                lower,
                upper,
                color="#d62728",
                alpha=0.20,
                linewidth=0,
                label=band_label,
            )

        ax.plot(
            t_eval,
            mean_values,
            color="black",
            linewidth=1.7,
            label="decoded mean",
        )
        ax.set_title(roi_cols[roi_idx], fontsize=8.5)
        ax.set_xlabel("Adaptive bin time")
        ax.set_ylabel("Decoded Tau SUVR")
        ax.grid(alpha=0.25)

    for ax in axes[n_plots:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels() if n_plots else ([], [])
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=24, loc="upper right")
    fig.suptitle(f"Decoded trajectory mean and variance: subtype {subtype+1}", fontsize=24)
    fig.tight_layout(rect=(0, 0, 0.985, 0.965))
    fig.savefig(out_dir / f"decoded_trajectory_subtype_{subtype}.png", dpi=600)
    plt.close(fig)


def _fit_projection_2d(x_fit: np.ndarray, args):
    """Fit UMAP if available; otherwise use a PCA fallback with the same transform API."""
    backend = args.trajectory_umap_backend
    if backend in {"auto", "umap"}:
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                random_state=args.seed,
            )
            reducer.fit(x_fit)
            return "umap", reducer
        except Exception as exc:
            if backend == "umap":
                raise RuntimeError(
                    "UMAP plotting requested, but fitting UMAP failed. Install with `pip install umap-learn` "
                    "or use --trajectory_umap_backend pca."
                ) from exc
            print(f"WARNING: UMAP unavailable or failed ({exc}). Falling back to PCA for trajectory projection.")

    # PCA fallback.
    x_mean = x_fit.mean(axis=0, keepdims=True)
    x_centered = x_fit - x_mean
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:2].T

    class PCATransformer:
        def __init__(self):
            self.components_ = components.T
            self.mean_ = x_mean.squeeze()

        def transform(self, x):
            return (np.asarray(x, dtype=np.float32) - x_mean) @ components

    return "pca", PCATransformer()


def _bin_label_map(df: pd.DataFrame, args) -> dict[int, str]:
    labels: dict[int, str] = {}
    if args.bin_col not in df.columns:
        return labels
    bin_values = numeric_series(df[args.bin_col])
    for b in sorted(bin_values.dropna().astype(int).unique().tolist()):
        m = bin_values.astype("Int64") == b
        if hasattr(args, "dynamics_label_col") and args.dynamics_label_col in df.columns:
            lab_vals = df.loc[m, args.dynamics_label_col].dropna().astype(str).unique().tolist()
            if lab_vals:
                labels[b] = lab_vals[0]
                continue
        if b == getattr(args, "stage0_bin_value", -999):
            labels[b] = getattr(args, "stage0_label", "S0")
            continue
        if args.stage_col in df.columns:
            stage_vals = numeric_series(df.loc[m, args.stage_col]).dropna().astype(int)
            if len(stage_vals):
                a, c = int(stage_vals.min()), int(stage_vals.max())
                labels[b] = f"S{a}" if a == c else f"S{a}-{c}"
                continue
        labels[b] = f"bin {b}"
    return labels


def plot_decoded_trajectories_on_umap(
    decoded_traj_by_subtype: dict[int, np.ndarray],
    df: pd.DataFrame,
    roi_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    args,
    out_dir: Path,
) -> None:
    """Project real ROI data and decoded trajectories onto a shared UMAP/PCA plane."""
    if not args.plot_dynamics_umap:
        return

    if args.bin_col not in df.columns:
        print(f"WARNING: cannot plot trajectory UMAP because {args.bin_col!r} is missing.")
        return

    x_raw_all = df[roi_cols].to_numpy(dtype=np.float32)
    x_std_all = (x_raw_all - mean.reshape(1, -1)) / std.reshape(1, -1)

    real_mask = numeric_series(df[args.bin_col]).notna().to_numpy(dtype=bool, copy=True)
    if args.use_col in df.columns and args.visualize_use_in_otfm_only:
        real_mask = real_mask & normalize_bool_series(df[args.use_col]).to_numpy(dtype=bool)
    if args.subtype_col in df.columns:
        real_mask = real_mask & numeric_series(df[args.subtype_col]).notna().to_numpy(dtype=bool)

    fit_idx = np.flatnonzero(real_mask)
    if len(fit_idx) < 3:
        print("WARNING: not enough real points for UMAP trajectory visualization.")
        return

    rng = np.random.default_rng(args.seed)
    if args.umap_max_fit_points > 0 and len(fit_idx) > args.umap_max_fit_points:
        fit_idx = rng.choice(fit_idx, size=args.umap_max_fit_points, replace=False)

    backend_name, reducer = _fit_projection_2d(x_std_all[fit_idx], args)
    real_coords = reducer.transform(x_std_all)

    coord_df = pd.DataFrame(
        {
            "_source_row": df["_source_row"].to_numpy() if "_source_row" in df.columns else np.arange(len(df)),
            "umap1": real_coords[:, 0],
            "umap2": real_coords[:, 1],
            "used_for_umap_display": real_mask,
        }
    )
    for col in [args.subtype_col, args.bin_col, args.stage_col, args.use_col, getattr(args, "dynamics_label_col", ""), "PTID", "SCANDATE", "Research Group"]:
        if col and col in df.columns:
            coord_df[col] = df[col].to_numpy()
    coord_df.to_csv(out_dir / f"{backend_name}_real_data_coordinates.csv", index=False)

    bin_labels = _bin_label_map(df.loc[real_mask].copy(), args)
    bin_values = numeric_series(df.loc[real_mask, args.bin_col]).astype(int).to_numpy()
    real_subtype = numeric_series(df.loc[real_mask, args.subtype_col]).to_numpy() if args.subtype_col in df.columns else np.full(real_mask.sum(), np.nan)
    real_stage = numeric_series(df.loc[real_mask, args.stage_col]).to_numpy() if args.stage_col in df.columns else np.full(real_mask.sum(), np.nan)
    real_plot_coords = real_coords[real_mask]

    fig, ax = plt.subplots(figsize=(14, 8))
    for b in sorted(np.unique(bin_values).astype(int).tolist()):
        m = bin_values == b
        label = bin_labels.get(b, f"bin {b}")
        ax.scatter(real_plot_coords[m, 0], real_plot_coords[m, 1], s=14, alpha=0.45, label=label)

    all_path_rows = []
    subtype_colors = plt.cm.tab10(np.linspace(0, 1, max(len(decoded_traj_by_subtype), 1)))
    min_bin = int(np.nanmin(bin_values))
    max_bin = int(np.nanmax(bin_values))

    for color_i, subtype in enumerate(sorted(decoded_traj_by_subtype)):
        x_traj = decoded_traj_by_subtype[subtype]
        T, N, D = x_traj.shape
        t_eval = np.linspace(min_bin, max_bin, T)
        x_std_traj = (x_traj.reshape(T * N, D) - mean.reshape(1, -1)) / std.reshape(1, -1)
        traj_coords = reducer.transform(x_std_traj).reshape(T, N, 2)
        mean_path = traj_coords.mean(axis=1)

        sample_count = min(args.n_umap_trajectory_samples, N)

        ax.plot(
            mean_path[:, 0],
            mean_path[:, 1],
            color=subtype_colors[color_i],
            linewidth=3.0,
            marker="o",
            markersize=3,
            label=f"decoded mean path subtype {subtype}",
        )
        ax.scatter(mean_path[0, 0], mean_path[0, 1], color=subtype_colors[color_i], s=70, marker="s")
        ax.scatter(mean_path[-1, 0], mean_path[-1, 1], color=subtype_colors[color_i], s=90, marker="*")

        for ti in range(T):
            all_path_rows.append(
                {
                    "sustain_subtype": subtype,
                    "branch_subtype": subtype,
                    "initial_source": "shared_root" if getattr(args, "shared_root_stage0", False) else "subtype_specific",
                    "time_index": ti,
                    "adaptive_time": float(t_eval[ti]),
                    "umap1_mean": float(mean_path[ti, 0]),
                    "umap2_mean": float(mean_path[ti, 1]),
                }
            )
        np.save(out_dir / f"{backend_name}_decoded_traj_coords_subtype_{subtype}.npy", traj_coords.astype(np.float32))

    pd.DataFrame(all_path_rows).to_csv(out_dir / f"{backend_name}_decoded_mean_path_by_subtype.csv", index=False)

    ax.set_title(f"{backend_name.upper()} projection for decoded trajectories", fontsize=18, fontweight="bold")
    ax.set_xlabel(f"{backend_name.upper()}1", fontsize=18, fontweight="bold")
    ax.set_ylabel(f"{backend_name.upper()}2", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=18, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{backend_name}_real_data_with_decoded_dynamics.png", dpi=600)
    plt.close(fig)


