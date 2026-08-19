#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AutoEncoder-latent Conditional OT-CFM training entry point.

This module owns CLI parsing, data preparation, model training, checkpoint saving,
and the end-to-end pipeline orchestration. Helper functions, models, and plotting
live in Utilities.py, Models.py, and Visualization.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.platform.startswith("win"):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from .models import (
        AutoEncoder,
        ConditionalVectorField,
        infer_autoencoder_architecture,
    )
    from .utilities import (
        LatentPool,
        add_stage0_extra_dynamics_bin,
        build_transitions,
        compute_scaler,
        decode_trajectory,
        effective_sample_size,
        euler_trajectory_latent,
        fill_missing_by_same_ptid_nearest_date,
        get_cortical_roi_cols,
        get_subtype_probability_cols,
        normalize_bool_series,
        numeric_series,
        sample_cfm_batch,
        sample_pool,
        save_json,
        set_seed,
    )
    from .visualization import (
        plot_encoded_latent_dynamics_on_umap,
        plot_decoded_trajectories,
        plot_decoded_trajectories_on_umap,
        visualize_encoded_latent_umap,
        visualize_latent,
    )
except ImportError:  # direct script execution from this folder
    from .models import (
        AutoEncoder,
        ConditionalVectorField,
        infer_autoencoder_architecture,
    )
    from .utilities import (
        LatentPool,
        add_stage0_extra_dynamics_bin,
        build_transitions,
        compute_scaler,
        decode_trajectory,
        effective_sample_size,
        euler_trajectory_latent,
        fill_missing_by_same_ptid_nearest_date,
        get_cortical_roi_cols,
        get_subtype_probability_cols,
        normalize_bool_series,
        numeric_series,
        sample_cfm_batch,
        sample_pool,
        save_json,
        set_seed,
    )
    from .visualization import (
        plot_encoded_latent_dynamics_on_umap,
        plot_decoded_trajectories,
        plot_decoded_trajectories_on_umap,
        visualize_encoded_latent_umap,
        visualize_latent,
    )

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "SuStaIn_Main" / "tau83_binning_comparison" / "outputs" / "scheme_B_fixed_5positive" / "01_binned_dataset_all_scans.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs_ae_latent_ot_cfm_bin_schemeB"


def prepare_dataframe(args, out_dir: Path):
    df = pd.read_csv(args.input_csv)
    expected_dim = None if args.expected_dim < 0 else args.expected_dim
    roi_cols = get_cortical_roi_cols(df, expected_dim=expected_dim, mode=args.roi_mode)
    prob_cols = get_subtype_probability_cols(df)

    numeric_cols = list(
        set(
            roi_cols
            + prob_cols
            + [
                args.stage_col,
                args.subtype_col,
                args.bin_col,
                getattr(args, "source_bin_col", args.bin_col),
            ]
        )
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = numeric_series(df[col])

    if args.use_col in df.columns:
        df[args.use_col] = normalize_bool_series(df[args.use_col])

    # Optional: stage 0 becomes an explicit dynamics bin, and positive bins are shifted by +1.
    df = add_stage0_extra_dynamics_bin(df, args, out_dir)

    before_missing = {
        "scope": "before_same_ptid_fill",
        "rows": int(len(df)),
        "rows_with_any_roi_nan": int((df[roi_cols].isna().sum(axis=1) > 0).sum()),
        "missing_roi_cells": int(df[roi_cols].isna().sum().sum()),
    }

    if args.fill_missing_same_ptid:
        df, fill_log = fill_missing_by_same_ptid_nearest_date(
            df,
            feature_cols=roi_cols,
            max_day_diff=args.max_fill_day_diff,
        )
        fill_log.to_csv(out_dir / "same_ptid_fill_log.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "same_ptid_fill_log.csv", index=False)

    after_missing = {
        "scope": "after_same_ptid_fill",
        "rows": int(len(df)),
        "rows_with_any_roi_nan": int((df[roi_cols].isna().sum(axis=1) > 0).sum()),
        "missing_roi_cells": int(df[roi_cols].isna().sum().sum()),
    }

    df = df.dropna(subset=roi_cols).copy()
    df = df.reset_index(drop=False).rename(columns={"index": "_source_row"})

    final_missing = {
        "scope": "after_drop_roi_nan",
        "rows": int(len(df)),
        "rows_with_any_roi_nan": int((df[roi_cols].isna().sum(axis=1) > 0).sum()),
        "missing_roi_cells": int(df[roi_cols].isna().sum().sum()),
    }

    pd.DataFrame([before_missing, after_missing, final_missing]).to_csv(
        out_dir / "missing_summary.csv",
        index=False,
    )
    pd.DataFrame(
        {"feature_index": np.arange(len(roi_cols)), "feature_name": roi_cols}
    ).to_csv(out_dir / "roi_columns_used.csv", index=False)

    return df, roi_cols, prob_cols


def train_autoencoder(df: pd.DataFrame, roi_cols: list[str], prob_cols: list[str], args, device: torch.device):
    x_raw = df[roi_cols].to_numpy(dtype=np.float32)
    mean, std = compute_scaler(x_raw)
    x = (x_raw - mean) / std

    # Labels for auxiliary losses. Missing labels get ignored.
    subtype = pd.to_numeric(df.get(args.subtype_col, pd.Series(np.nan, index=df.index)), errors="coerce") # type: ignore
    bin_label = pd.to_numeric(df.get(args.bin_col, pd.Series(np.nan, index=df.index)), errors="coerce") # type: ignore
    n_subtypes = len(prob_cols)
    n_bins = int(np.nanmax(bin_label.to_numpy())) + 1 if np.isfinite(bin_label.to_numpy()).any() else args.n_bins

    x_t = torch.from_numpy(x).float().to(device)
    subtype_t = torch.from_numpy(
        subtype.fillna(-1).astype(int).to_numpy(copy=True)
    ).long().to(device)
    bin_t = torch.from_numpy(
        bin_label.fillna(-1).astype(int).to_numpy(copy=True)
    ).long().to(device)
    subtype_mask = subtype_t >= 0
    bin_mask = bin_t >= 0

    model = AutoEncoder(
        input_dim=len(roi_cols),
        latent_dim=args.latent_dim,
        hidden_width=args.ae_hidden_width,
        n_subtypes=n_subtypes,
        n_bins=max(n_bins, args.n_bins),
        dropout=args.ae_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.ae_lr,
        weight_decay=args.ae_weight_decay,
    )
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    n = x_t.shape[0]
    loss_rows = []
    rng = np.random.default_rng(args.seed)

    model.train()
    for epoch in range(args.ae_epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_sub = 0.0
        epoch_bin = 0.0
        n_batches = 0

        for start in range(0, n, args.ae_batch_size):
            idx_np = perm[start:start + args.ae_batch_size]
            idx = torch.as_tensor(idx_np, device=device)
            xb = x_t[idx]

            optimizer.zero_grad(set_to_none=True)
            x_hat, z, subtype_logits, bin_logits = model(xb)
            rec_loss = mse(x_hat, xb)

            sub_loss = torch.zeros((), device=device)
            if args.subtype_loss_weight > 0:
                sm = subtype_mask[idx]
                if sm.any():
                    sub_loss = ce(subtype_logits[sm], subtype_t[idx][sm])

            b_loss = torch.zeros((), device=device)
            if args.bin_loss_weight > 0:
                bm = bin_mask[idx]
                if bm.any():
                    b_loss = ce(bin_logits[bm], bin_t[idx][bm])

            # A mild latent L2 penalty keeps the latent scale stable for OT.
            latent_l2 = torch.mean(z ** 2)
            loss = (
                rec_loss
                + args.subtype_loss_weight * sub_loss
                + args.bin_loss_weight * b_loss
                + args.latent_l2_weight * latent_l2
            )

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_rec += float(rec_loss.detach().cpu())
            epoch_sub += float(sub_loss.detach().cpu())
            epoch_bin += float(b_loss.detach().cpu())
            n_batches += 1

        if epoch % args.ae_log_interval == 0 or epoch == args.ae_epochs - 1:
            row = {
                "epoch": epoch,
                "loss": epoch_loss / max(n_batches, 1),
                "reconstruction_mse": epoch_rec / max(n_batches, 1),
                "subtype_ce": epoch_sub / max(n_batches, 1),
                "bin_ce": epoch_bin / max(n_batches, 1),
            }
            print(
                f"AE epoch={epoch:04d} loss={row['loss']:.6e} "
                f"rec={row['reconstruction_mse']:.6e} "
                f"sub_ce={row['subtype_ce']:.4f} bin_ce={row['bin_ce']:.4f}"
            )
            loss_rows.append(row)

    model.eval()
    with torch.no_grad():
        x_hat, z, subtype_logits, bin_logits = model(x_t)
        rec_mse_by_row = torch.mean((x_hat - x_t) ** 2, dim=1).detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        x_hat_std_np = x_hat.detach().cpu().numpy()
        x_hat_raw_np = x_hat_std_np * std + mean

    ae_summary = {
        "n_rows_ae": int(len(df)),
        "input_dim": int(len(roi_cols)),
        "latent_dim": int(args.latent_dim),
        "mean_reconstruction_mse_standardized": float(np.mean(rec_mse_by_row)),
        "median_reconstruction_mse_standardized": float(np.median(rec_mse_by_row)),
        "p95_reconstruction_mse_standardized": float(np.percentile(rec_mse_by_row, 95)),
    }

    return model, z_np, x_hat_raw_np, rec_mse_by_row, mean, std, pd.DataFrame(loss_rows), ae_summary


def build_latent_pools(df: pd.DataFrame, z_np: np.ndarray, prob_cols: list[str], args, device: torch.device):
    required = [args.subtype_col, args.bin_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns for latent FM: {missing}")

    work = df.copy()
    if args.use_col in work.columns and args.require_use_in_otfm:
        work = work[normalize_bool_series(work[args.use_col])].copy()

    shared_root_stage0 = bool(getattr(args, "shared_root_stage0", False))
    root_bin = int(getattr(args, "root_bin_value", 0))
    start_bin = int(getattr(args, "subtype_specific_start_bin", root_bin + 1))

    bin_series = pd.to_numeric(work[args.bin_col], errors="coerce")
    subtype_series = pd.to_numeric(work[args.subtype_col], errors="coerce")
    if shared_root_stage0 and args.stage_col in work.columns:
        stage_series = pd.to_numeric(work[args.stage_col], errors="coerce")
        root_row_mask = (bin_series == root_bin) | (stage_series == getattr(args, "stage0_raw_value", 0))
    else:
        stage_series = pd.Series(np.nan, index=work.index, dtype=float)
        root_row_mask = bin_series == root_bin

    if shared_root_stage0:
        valid_rows = root_row_mask | (subtype_series.notna() & bin_series.notna())
    else:
        valid_rows = subtype_series.notna() & bin_series.notna()
    work = work[valid_rows].copy()

    # z_np is aligned to original df positions; use row positions in df.
    row_positions = work.index.to_numpy()
    subtype_num = pd.to_numeric(work[args.subtype_col], errors="coerce")
    bin_num = pd.to_numeric(work[args.bin_col], errors="coerce")
    stage_num = pd.to_numeric(work[args.stage_col], errors="coerce") if args.stage_col in work.columns else pd.Series(np.nan, index=work.index)
    subtype_arr = subtype_num.to_numpy(dtype=float)
    bin_arr = bin_num.to_numpy(dtype=float)

    pools: dict[tuple[int | str, int], LatentPool] = {}
    rows = []
    diagnostic_rows = []

    root_mask = (bin_num == root_bin).to_numpy(dtype=bool)
    if shared_root_stage0 and args.stage_col in work.columns:
        root_mask = root_mask | (stage_num == getattr(args, "stage0_raw_value", 0)).to_numpy(dtype=bool)

    if shared_root_stage0:
        idx_local = np.flatnonzero(root_mask)
        if len(idx_local) < args.min_pool_size:
            raise ValueError(
                f"Shared root bin {root_bin} has only {len(idx_local)} samples. "
                f"Need at least --min_pool_size={args.min_pool_size}."
            )
        idx_df = row_positions[idx_local]
        raw_weights = np.full(len(idx_local), 1.0 / len(idx_local), dtype=np.float32)
        root_pool = LatentPool(
            subtype="shared",
            bin_index=root_bin,
            z=torch.from_numpy(z_np[idx_df].astype(np.float32)).float().to(device),
            weights=torch.from_numpy(raw_weights).float().to(device),
            row_indices=df["_source_row"].iloc[idx_df].to_numpy(),
            effective_n=effective_sample_size(raw_weights),
            mean_posterior=float("nan"),
            n=int(len(idx_local)),
        )
        pools[("shared", root_bin)] = root_pool
        rows.append(
            {
                "sustain_subtype": "shared",
                "coarse_stage_bin": root_bin,
                "n_candidates": root_pool.n,
                "effective_n": root_pool.effective_n,
                "mean_posterior": np.nan,
                "posterior_power": args.posterior_power,
                "min_posterior": args.min_posterior,
                "pool_assignment": "shared_root",
                "is_shared_root": True,
            }
        )
        diagnostic_rows.append(
            {
                "record_type": "shared_root",
                "sustain_subtype": "shared",
                "dynamics_bin": root_bin,
                "n_samples": int(len(idx_local)),
                "is_shared_root": True,
                "warning": "",
            }
        )

    valid_branch = subtype_num.notna() & bin_num.notna()
    if shared_root_stage0:
        valid_branch = valid_branch & (bin_num >= start_bin)
    subtype_ids = sorted(subtype_num.loc[valid_branch].dropna().astype(int).unique().tolist())
    bin_ids = sorted(bin_num.loc[valid_branch].dropna().astype(int).unique().tolist())

    for subtype in subtype_ids:
        prob_col = f"subtype_{subtype}_prob"
        if prob_col not in prob_cols:
            raise ValueError(f"Missing posterior column for subtype {subtype}: {prob_col}")

        posterior_all = pd.to_numeric(work[prob_col], errors="coerce").to_numpy(dtype=float)

        for b in bin_ids:
            if shared_root_stage0 and int(b) < start_bin:
                continue
            if args.pool_assignment == "hard":
                mask = (subtype_arr == subtype) & (bin_arr == b)
            elif args.pool_assignment == "same_bin_soft_subtype":
                mask = bin_arr == b
            else:
                raise ValueError(f"Unknown --pool_assignment {args.pool_assignment!r}")

            mask = mask & np.isfinite(posterior_all) & (posterior_all >= args.min_posterior)
            idx_local = np.flatnonzero(mask)
            if len(idx_local) == 0 and shared_root_stage0:
                msg = f"Subtype {subtype}, bin {b} has no target samples; skipping this branch pool."
                print("WARNING:", msg)
                diagnostic_rows.append(
                    {
                        "record_type": "empty_target_bin",
                        "sustain_subtype": subtype,
                        "dynamics_bin": int(b),
                        "n_samples": 0,
                        "is_shared_root": False,
                        "warning": msg,
                    }
                )
                continue
            if len(idx_local) < args.min_pool_size:
                msg = (
                    f"Subtype {subtype}, bin {b} has only {len(idx_local)} samples after filtering. "
                    f"Reduce --min_pool_size/--min_posterior or use --pool_assignment same_bin_soft_subtype."
                )
                if args.skip_small_pools:
                    print("WARNING:", msg)
                    diagnostic_rows.append(
                        {
                            "record_type": "small_branch_pool_skipped",
                            "sustain_subtype": subtype,
                            "dynamics_bin": int(b),
                            "n_samples": int(len(idx_local)),
                            "is_shared_root": False,
                            "warning": msg,
                        }
                    )
                    continue
                raise ValueError(msg)

            idx_df = row_positions[idx_local]
            raw_weights = np.power(np.clip(posterior_all[idx_local], 0.0, 1.0), args.posterior_power)
            raw_weights = raw_weights / max(raw_weights.sum(), 1e-12)

            z_pool = z_np[idx_df].astype(np.float32)
            pool = LatentPool(
                subtype=subtype,
                bin_index=int(b),
                z=torch.from_numpy(z_pool).float().to(device),
                weights=torch.from_numpy(raw_weights.astype(np.float32)).float().to(device),
                row_indices=df["_source_row"].iloc[idx_df].to_numpy(),
                effective_n=effective_sample_size(raw_weights),
                mean_posterior=float(np.mean(posterior_all[idx_local])),
                n=int(len(idx_local)),
            )
            pools[(subtype, int(b))] = pool
            rows.append(
                {
                    "sustain_subtype": subtype,
                    "coarse_stage_bin": int(b),
                    "n_candidates": pool.n,
                    "effective_n": pool.effective_n,
                    "mean_posterior": pool.mean_posterior,
                    "posterior_power": args.posterior_power,
                    "min_posterior": args.min_posterior,
                    "pool_assignment": args.pool_assignment,
                    "is_shared_root": False,
                }
            )
            diagnostic_rows.append(
                {
                    "record_type": "branch_pool",
                    "sustain_subtype": subtype,
                    "dynamics_bin": int(b),
                    "n_samples": int(len(idx_local)),
                    "is_shared_root": False,
                    "warning": "",
                }
            )

    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "latent_pool_summary.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(Path(args.out_dir) / "shared_root_diagnostics.csv", index=False)
    if not pools:
        raise ValueError("No latent pools were built.")
    return pools


def train_latent_ot_cfm(pools, n_subtypes: int, latent_dim: int, args, device: torch.device):
    transitions = build_transitions(pools, args=args)
    model = ConditionalVectorField(
        dim=latent_dim,
        n_subtypes=n_subtypes,
        hidden_width=args.cfm_hidden_width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.cfm_lr, weight_decay=args.cfm_weight_decay)

    loss_rows = []
    model.train()
    for step in range(args.cfm_iters):
        optimizer.zero_grad(set_to_none=True)
        t, zt, ut, cond, loss_weights = sample_cfm_batch( # type: ignore
            transitions=transitions,
            n_subtypes=n_subtypes,
            batch_size=args.cfm_batch_size,
            sigma=args.cfm_sigma,
            device=device,
            return_loss_weights=True,
        )
        vt = model(zt, t, cond)
        per_sample_loss = torch.mean((vt - ut) ** 2, dim=1)
        loss = torch.sum(per_sample_loss * loss_weights) / torch.clamp(loss_weights.sum(), min=1e-8)
        loss.backward()
        optimizer.step()

        if step % args.cfm_log_interval == 0 or step == args.cfm_iters - 1:
            loss_value = float(loss.detach().cpu())
            print(f"CFM iter={step:05d} loss={loss_value:.6e}")
            loss_rows.append({"iter": step, "loss": loss_value})

    return model, pd.DataFrame(loss_rows)


def decoded_sample_trajectories_to_dataframe(
    x_traj: np.ndarray,
    roi_cols: list[str],
    adaptive_time: np.ndarray,
    n_trajectories: int,
) -> pd.DataFrame:
    x = np.asarray(x_traj, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("x_traj must have shape T x N x ROI.")

    n_time, n_available, n_roi = x.shape
    if n_roi != len(roi_cols):
        raise ValueError(
            f"x_traj has {n_roi} ROI columns, but roi_cols contains {len(roi_cols)} names."
        )

    adaptive = np.asarray(adaptive_time, dtype=float)
    if adaptive.shape[0] != n_time:
        raise ValueError(
            f"adaptive_time length {adaptive.shape[0]} does not match trajectory time length {n_time}."
        )

    n_export = min(max(int(n_trajectories), 0), n_available)
    columns = ["trajectory_id", "time_index", "adaptive_time"] + list(roi_cols)
    if n_export == 0:
        return pd.DataFrame(columns=columns)

    parts = []
    for trajectory_id in range(n_export):
        part = pd.DataFrame(x[:, trajectory_id, :], columns=roi_cols)
        part.insert(0, "adaptive_time", adaptive)
        part.insert(0, "time_index", np.arange(n_time, dtype=int))
        part.insert(0, "trajectory_id", trajectory_id)
        parts.append(part)

    return pd.concat(parts, ignore_index=True)[columns]


def simulate_decode_and_plot(autoencoder, cfm_model, pools, df, roi_cols, mean, std, prob_cols, args, device, z_np=None, encoded_projection=None):
    out_dir = Path(args.out_dir)
    n_subtypes = len(prob_cols)
    shared_root_stage0 = bool(getattr(args, "shared_root_stage0", False))
    root_bin = int(getattr(args, "root_bin_value", 0))
    branch_subtypes = sorted({int(s) for s, _ in pools if s != "shared"})
    max_bin = max(b for s, b in pools.keys() if s != "shared")
    min_bin = root_bin if shared_root_stage0 else min(b for s, b in pools.keys() if s != "shared")
    t_eval = torch.linspace(float(min_bin), float(max_bin), args.n_eval_steps, device=device)
    t_eval_np = t_eval.detach().cpu().numpy()
    np.save(out_dir / "latent_cfm_t_eval.npy", t_eval_np)

    x_raw = df[roi_cols].to_numpy(dtype=np.float32)
    clip_low = clip_high = None
    if args.clip_decoded_percentile > 0:
        q = args.clip_decoded_percentile
        clip_low = np.percentile(x_raw, q, axis=0).astype(np.float32)
        clip_high = np.percentile(x_raw, 100 - q, axis=0).astype(np.float32)

    traj_summary_rows = []
    decoded_traj_by_subtype: dict[int, np.ndarray] = {}
    latent_traj_by_subtype: dict[int, np.ndarray] = {}

    shared_z0 = None
    shared_n0 = 0
    if shared_root_stage0:
        root_key = ("shared", root_bin)
        if root_key not in pools:
            raise ValueError(f"Shared root pool {root_key!r} is missing.")
        root_pool = pools[root_key]
        shared_n0 = min(args.n_initial_samples, root_pool.n)
        if getattr(args, "same_root_samples_for_all_subtypes", True):
            shared_z0 = sample_pool(root_pool, shared_n0)

    for subtype in branch_subtypes:
        subtype_bins = sorted(b for s, b in pools if s == subtype)
        first_bin = subtype_bins[0]
        if shared_root_stage0:
            pool0 = pools[("shared", root_bin)]
            n0 = shared_n0
            z0 = shared_z0.clone() if shared_z0 is not None else sample_pool(pool0, n0)
            initial_source = "shared_root"
        else:
            pool0 = pools[(subtype, first_bin)]
            n0 = min(args.n_initial_samples, pool0.n)
            z0 = sample_pool(pool0, n0)
            initial_source = "subtype_specific"

        z_traj = euler_trajectory_latent(
            model=cfm_model,
            z0=z0,
            subtype=subtype,
            n_subtypes=n_subtypes,
            t_eval=t_eval,
            device=device,
            latent_velocity_clip=args.latent_velocity_clip,
        )
        z_traj_np = z_traj.detach().cpu().numpy()
        latent_traj_by_subtype[int(subtype)] = z_traj_np
        np.save(out_dir / f"latent_traj_subtype_{subtype}.npy", z_traj_np)

        x_traj = decode_trajectory(
            autoencoder=autoencoder,
            z_traj=z_traj,
            mean=mean.squeeze(),
            std=std.squeeze(),
            device=device,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        decoded_traj_by_subtype[int(subtype)] = x_traj
        np.save(out_dir / f"decoded_suvr_traj_subtype_{subtype}.npy", x_traj)

        # Save mean and median decoded trajectories in ROI space for BrainNet/NIfTI visualization.
        mean_traj = x_traj.mean(axis=1)  # T x ROI
        median_traj = np.median(x_traj, axis=1)
        mean_df = pd.DataFrame(mean_traj, columns=roi_cols)
        median_df = pd.DataFrame(median_traj, columns=roi_cols)
        mean_df.insert(0, "branch_subtype", subtype)
        mean_df.insert(0, "initial_source", initial_source)
        mean_df.insert(0, "adaptive_time", t_eval_np)
        median_df.insert(0, "branch_subtype", subtype)
        median_df.insert(0, "initial_source", initial_source)
        median_df.insert(0, "adaptive_time", t_eval_np)
        mean_df.to_csv(out_dir / f"decoded_mean_roi_trajectory_subtype_{subtype}.csv", index=False)
        median_df.to_csv(out_dir / f"decoded_median_roi_trajectory_subtype_{subtype}.csv", index=False)

        sample_csv_name = ""
        if getattr(args, "n_decoded_csv_trajectories", 0) > 0:
            sample_csv_name = f"decoded_sample_roi_trajectories_subtype_{subtype}.csv"
            sample_df = decoded_sample_trajectories_to_dataframe(
                x_traj=x_traj,
                roi_cols=roi_cols,
                adaptive_time=t_eval_np,
                n_trajectories=args.n_decoded_csv_trajectories,
            )
            sample_df.to_csv(out_dir / sample_csv_name, index=False)

        plot_decoded_trajectories(
            subtype=subtype,
            pools=pools,
            roi_cols=roi_cols,
            x_traj=x_traj,
            df=df,
            roi_mean=mean,
            roi_std=std,
            args=args,
            out_dir=out_dir,
        )

        traj_summary_rows.append(
            {
                "sustain_subtype": subtype,
                "branch_subtype": subtype,
                "initial_source": initial_source,
                "n_initial_samples": int(n0),
                "first_bin": int(root_bin if shared_root_stage0 else first_bin),
                "last_bin": int(subtype_bins[-1]),
                "decoded_traj_path": f"decoded_suvr_traj_subtype_{subtype}.npy",
                "decoded_mean_roi_trajectory_csv": f"decoded_mean_roi_trajectory_subtype_{subtype}.csv",
                "decoded_median_roi_trajectory_csv": f"decoded_median_roi_trajectory_subtype_{subtype}.csv",
                "decoded_sample_roi_trajectories_csv": sample_csv_name,
            }
        )

    pd.DataFrame(traj_summary_rows).to_csv(out_dir / "decoded_trajectory_summary.csv", index=False)

    plot_decoded_trajectories_on_umap(
        decoded_traj_by_subtype=decoded_traj_by_subtype,
        df=df,
        roi_cols=roi_cols,
        mean=mean.squeeze(),
        std=std.squeeze(),
        args=args,
        out_dir=out_dir,
    )

    # if z_np is not None:
    #     plot_encoded_latent_dynamics_on_umap(
    #         latent_traj_by_subtype=latent_traj_by_subtype,
    #         cfm_model=cfm_model,
    #         df=df,
    #         z_np=z_np,
    #         prob_cols=prob_cols,
    #         args=args,
    #         out_dir=out_dir,
    #         device=device,
    #         t_eval=t_eval_np,
    #         projection=encoded_projection,
    #     )

    return decoded_traj_by_subtype


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}.")


def should_train(args) -> bool:
    return getattr(args, "pipeline_mode", "train_and_visualize") in {"train_and_visualize", "train_only"}


def should_visualize(args) -> bool:
    return getattr(args, "pipeline_mode", "train_and_visualize") in {"train_and_visualize", "visualize_only"}


def resolve_trained_artifacts_dir(args) -> Path:
    raw_dir = getattr(args, "trained_artifacts_dir", "") or getattr(args, "out_dir", "")
    return Path(raw_dir).expanduser().resolve()


def load_torch_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_autoencoder_checkpoint(artifacts_dir: Path, device: torch.device):
    ckpt_path = artifacts_dir / "autoencoder_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing trained autoencoder checkpoint: {ckpt_path}")
    ckpt = load_torch_checkpoint(ckpt_path, device)
    architecture = infer_autoencoder_architecture(ckpt)
    model = AutoEncoder(
        input_dim=architecture["input_dim"],
        latent_dim=architecture["latent_dim"],
        hidden_width=architecture["hidden_width"],
        n_subtypes=architecture["n_subtypes"],
        n_bins=architecture["n_bins"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["autoencoder_state_dict"])
    model.eval()
    return model, ckpt


def load_cfm_checkpoint(artifacts_dir: Path, device: torch.device):
    ckpt_path = artifacts_dir / "latent_ot_cfm_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing trained CFM checkpoint: {ckpt_path}")
    ckpt = load_torch_checkpoint(ckpt_path, device)
    model = ConditionalVectorField(
        dim=int(ckpt["latent_dim"]),
        n_subtypes=int(ckpt["n_subtypes"]),
        hidden_width=int(ckpt["cfm_hidden_width"]),
    ).to(device)
    model.load_state_dict(ckpt["cfm_state_dict"])
    model.eval()
    return model, ckpt


def _latent_sort_key(col: str) -> int:
    return int(col[1:]) if col.startswith("z") and col[1:].isdigit() else 10_000


def load_or_encode_latents_for_visualization(
    artifacts_dir: Path,
    df: pd.DataFrame,
    roi_cols: list[str],
    autoencoder,
    autoencoder_ckpt: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(autoencoder_ckpt["roi_mean"], dtype=np.float32)
    std = np.asarray(autoencoder_ckpt["roi_std"], dtype=np.float32)
    latent_csv = artifacts_dir / "encoded_latent_all_rows.csv"
    if latent_csv.exists():
        latent_df = pd.read_csv(latent_csv)
        z_cols = sorted([c for c in latent_df.columns if c.startswith("z") and c[1:].isdigit()], key=_latent_sort_key)
        if z_cols and "_source_row" in latent_df.columns and "_source_row" in df.columns:
            indexed = latent_df.set_index("_source_row")
            wanted_rows = df["_source_row"].to_numpy()
            if np.isin(wanted_rows, indexed.index.to_numpy()).all():
                aligned = indexed.loc[wanted_rows]
                z_np = aligned[z_cols].to_numpy(dtype=np.float32)
                if "reconstruction_mse_standardized" in aligned.columns:
                    rec_mse = aligned["reconstruction_mse_standardized"].to_numpy(dtype=np.float32)
                else:
                    rec_mse = np.full(len(df), np.nan, dtype=np.float32)
                return z_np, rec_mse, mean, std

    x_raw = df[roi_cols].to_numpy(dtype=np.float32)
    x_std = (x_raw - mean) / std
    x_t = torch.from_numpy(x_std).float().to(device)
    autoencoder.eval()
    with torch.no_grad():
        x_hat, z, _, _ = autoencoder(x_t)
    z_np = z.detach().cpu().numpy().astype(np.float32)
    rec_mse = torch.mean((x_hat - x_t) ** 2, dim=1).detach().cpu().numpy().astype(np.float32)
    return z_np, rec_mse, mean, std


def run_visualization_only(args, out_dir: Path, device: torch.device) -> None:
    artifacts_dir = resolve_trained_artifacts_dir(args)
    print(f"Visualization-only mode. Loading trained artifacts from: {artifacts_dir}")

    df, roi_cols, prob_cols = prepare_dataframe(args, out_dir)
    autoencoder, ae_ckpt = load_autoencoder_checkpoint(artifacts_dir, device)
    z_np, rec_mse, mean, std = load_or_encode_latents_for_visualization(
        artifacts_dir=artifacts_dir,
        df=df,
        roi_cols=roi_cols,
        autoencoder=autoencoder,
        autoencoder_ckpt=ae_ckpt,
        device=device,
    )
    cfm_model, cfm_ckpt = load_cfm_checkpoint(artifacts_dir, device)

    _, sep_df = visualize_latent(df, z_np, rec_mse, args, out_dir)
    _, encoded_umap_sep_df, encoded_projection = visualize_encoded_latent_umap( # type: ignore
        df,
        z_np,
        rec_mse,
        args,
        out_dir,
        return_projection=True,
    )

    pools = build_latent_pools(
        df=df,
        z_np=z_np,
        prob_cols=prob_cols,
        args=args,
        device=device,
    )

    trajectory_count = 0
    if not args.no_simulate:
        decoded = simulate_decode_and_plot(
            autoencoder=autoencoder,
            cfm_model=cfm_model,
            pools=pools,
            df=df,
            roi_cols=roi_cols,
            mean=mean,
            std=std,
            prob_cols=prob_cols,
            args=args,
            device=device,
            z_np=z_np,
            encoded_projection=encoded_projection,
        )
        trajectory_count = int(sum(arr.shape[1] for arr in decoded.values()))

    config = vars(args).copy()
    config.update(
        {
            "pipeline_mode": "visualize_only",
            "trained_artifacts_dir": str(artifacts_dir),
            "n_rows_after_roi_cleaning": int(len(df)),
            "n_roi_features": int(len(roi_cols)),
            "n_subtypes": int(cfm_ckpt.get("n_subtypes", len(prob_cols))),
            "latent_separation_summary": sep_df.to_dict(orient="records"),
            "encoded_latent_umap_summary": encoded_umap_sep_df.to_dict(orient="records"),
            "decoded_trajectory_count": trajectory_count,
        }
    )
    save_json(config, out_dir / "visualization_only_config.json")
    print(f"Visualization-only run complete. Outputs saved to: {out_dir}")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train autoencoder latent SuStaIn-conditional OT-CFM and decode trajectories to ROI SUVR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--pipeline_mode", default="visualize_only", choices=["train_and_visualize", "train_only", "visualize_only"], help="Run full pipeline, train and save artifacts only, or load existing artifacts and regenerate visualizations only.")
    parser.add_argument("--trained_artifacts_dir", default="", help="Directory containing saved model/artifact files for --pipeline_mode visualize_only. Defaults to --out_dir.")

    parser.add_argument("--roi_mode", default="ctx_lh_rh", choices=["ctx_lh_rh", "all_suvr"])
    parser.add_argument("--expected_dim", type=int, default=68, help="Set -1 to disable dimension check.")
    parser.add_argument("--stage_col", default="sustain_stage")
    parser.add_argument("--subtype_col", default="sustain_subtype")
    parser.add_argument("--bin_col", default="dynamics_bin", help="Dynamics-time bin column used for AE auxiliary loss and CFM pools. With --include_stage0_as_bin, this is created automatically.")
    parser.add_argument("--source_bin_col", default="coarse_stage_bin", help="Positive-stage coarse bin column to shift by +1 when stage 0 is inserted as dynamics bin 0.")
    parser.add_argument("--use_col", default="use_in_otfm")
    parser.add_argument("--require_use_in_otfm", action="store_true", default=True)
    parser.add_argument("--no_require_use_in_otfm", dest="require_use_in_otfm", action="store_false")
    parser.add_argument("--visualize_use_in_otfm_only", action="store_true", default=True)
    parser.add_argument("--fill_missing_same_ptid", action="store_true", default=True)
    parser.add_argument("--no_fill_missing_same_ptid", dest="fill_missing_same_ptid", action="store_false")
    parser.add_argument("--max_fill_day_diff", type=int, default=None)

    # Stage-0 handling. By default, create dynamics_bin=0 for sustain_stage==0,
    # and shift positive coarse bins to 1..K.
    parser.add_argument("--include_stage0_as_bin", action="store_true", default=True)
    parser.add_argument("--no_include_stage0_as_bin", dest="include_stage0_as_bin", action="store_false")
    parser.add_argument("--stage0_raw_value", type=int, default=0)
    parser.add_argument("--stage0_bin_value", type=int, default=0)
    parser.add_argument("--positive_bin_shift", type=int, default=1)
    parser.add_argument("--stage0_label", default="S0")
    parser.add_argument("--dynamics_label_col", default="dynamics_stage_group_label")

    # AE
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--ae_hidden_width", type=int, default=128)
    parser.add_argument("--ae_dropout", type=float, default=0.0)
    parser.add_argument("--ae_epochs", type=int, default=1000)
    parser.add_argument("--ae_batch_size", type=int, default=128)
    parser.add_argument("--ae_lr", type=float, default=1e-3)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-4)
    parser.add_argument("--ae_log_interval", type=int, default=50)
    parser.add_argument("--subtype_loss_weight", type=float, default=0.05)
    parser.add_argument("--bin_loss_weight", type=float, default=0.05)
    parser.add_argument("--latent_l2_weight", type=float, default=1e-4)

    # Latent CFM
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--pool_assignment", default="same_bin_soft_subtype", choices=["hard", "same_bin_soft_subtype"])
    parser.add_argument("--posterior_power", type=float, default=1.0)
    parser.add_argument("--min_posterior", type=float, default=0.1)
    parser.add_argument("--min_pool_size", type=int, default=5)
    parser.add_argument("--skip_small_pools", action="store_true")
    parser.add_argument("--shared_root_stage0", nargs="?", const=True, default=True, type=parse_bool, help="Treat root/bin0 or stage0 rows as one shared source distribution instead of subtype-specific bin0 pools.")
    parser.add_argument("--no_shared_root_stage0", dest="shared_root_stage0", action="store_false")
    parser.add_argument("--root_bin_value", type=int, default=0, help="Dynamics bin used as the shared root source when --shared_root_stage0 is true.")
    parser.add_argument("--subtype_specific_start_bin", type=int, default=1, help="First bin where subtype-specific branch pools begin.")
    parser.add_argument("--same_root_samples_for_all_subtypes", nargs="?", const=True, default=True, type=parse_bool, help="Use the same sampled root initial points for every subtype trajectory.")
    parser.add_argument("--no_same_root_samples_for_all_subtypes", dest="same_root_samples_for_all_subtypes", action="store_false")
    parser.add_argument("--root_loss_weight", type=float, default=1.0, help="Total loss weight assigned to shared-root-to-subtype transitions.")
    parser.add_argument("--branch_loss_weight", type=float, default=1.0, help="Loss weight assigned to each within-subtype transition.")
    parser.add_argument("--cfm_iters", type=int, default=5000)
    parser.add_argument("--cfm_batch_size", type=int, default=32)
    parser.add_argument("--cfm_lr", type=float, default=1e-4)
    parser.add_argument("--cfm_weight_decay", type=float, default=1e-4)
    parser.add_argument("--cfm_sigma", type=float, default=0.005)
    parser.add_argument("--cfm_hidden_width", type=int, default=256)
    parser.add_argument("--cfm_log_interval", type=int, default=100)
    parser.add_argument("--latent_velocity_clip", type=float, default=0.0)

    # Simulation and plotting
    parser.add_argument("--n_eval_steps", type=int, default=120)
    parser.add_argument("--n_initial_samples", type=int, default=32)
    parser.add_argument("--n_decoded_csv_trajectories", type=int, default=20, help="Number of decoded sample trajectories per subtype to export as long-form ROI CSV files. 0 disables sample CSV export.")
    parser.add_argument("--n_plot_rois", type=int, default=9)
    parser.add_argument("--n_overlay_traj", type=int, default=25)
    parser.add_argument("--plot_all_decoded_rois", nargs="?", const=True, default=True, type=parse_bool, help="Plot every ROI in decoded trajectory summary figures instead of randomly selecting --n_plot_rois.")
    parser.add_argument("--no_plot_all_decoded_rois", dest="plot_all_decoded_rois", action="store_false")
    parser.add_argument("--decoded_plot_n_rows", type=int, default=0, help="Rows for decoded mean/variance ROI grid. 0 chooses automatically and expands to fit all plotted ROIs.")
    parser.add_argument("--decoded_plot_n_cols", type=int, default=9, help="Columns for decoded mean/variance ROI grid.")
    parser.add_argument("--decoded_uncertainty_mode", default="std", choices=["std", "variance", "sem", "none"], help="Uncertainty band around decoded mean: standard deviation, variance, standard error, or no band.")
    parser.add_argument("--clip_decoded_percentile", type=float, default=0.0)
    parser.add_argument("--trajectory_init_mode", default="shared_root_anchor", choices=["shared_root_anchor", "shared_root_samples", "subtype_specific_early"], help="Initialization policy for plotted trajectories.")
    parser.add_argument("--root_anchor_mode", default="latent_medoid", choices=["latent_medoid", "latent_mean", "umap_medoid"], help="Shared-root anchor used by direct 2D trajectory visualizations.")
    parser.add_argument("--n_root_samples_for_plot", type=int, default=8, help="Maximum shared-root sample paths to draw when sample-path uncertainty is requested.")
    parser.add_argument("--plot_all_root_sample_paths", nargs="?", const=True, default=False, type=parse_bool, help="Explicitly draw every shared-root sample path. Disabled by default to avoid overplotting root samples.")
    parser.add_argument("--branch_anchor_mode", default="latent_medoid", choices=["latent_medoid", "latent_mean", "umap_medoid"], help="Subtype stage1 anchor mode for direct 2D shared-root bifurcation plots.")
    parser.add_argument("--n_root_highlight_points", type=int, default=20, help="Number of representative shared-root points to highlight in direct 2D bifurcation plots.")
    parser.add_argument("--root_highlight_mode", default="medoid_neighbors", choices=["random", "medoid_neighbors", "kmeans_centers"], help="How to choose highlighted shared-root representative points.")
    parser.add_argument("--n_stage1plus_velocity_arrows_per_subtype", type=int, default=30, help="Maximum observed-point velocity arrows per subtype for bin >= subtype_specific_start_bin.")
    parser.add_argument("--stage1plus_velocity_step", type=float, default=0.3, help="Display step h for UMAP2D_DIRECT stage1+ local velocity arrows. This affects arrow length in the plot, not model training.")
    parser.add_argument("--stage1plus_velocity_arrow_display_scale", type=float, default=1.0, help="Extra display-only multiplier for UMAP2D_DIRECT stage1+ velocity arrows after clipping/normalization.")
    parser.add_argument("--stage1plus_marker_subtype0", default="o", help="Marker for subtype 0 stage1+ velocity source points.")
    parser.add_argument("--stage1plus_marker_subtype1", default="h", help="Marker for subtype 1 stage1+ velocity source points.")
    parser.add_argument("--plot_root_to_stage1_arrows", nargs="?", const=True, default=True, type=parse_bool, help="Draw thick shared-root-to-stage1 bifurcation arrows.")
    parser.add_argument("--trajectory_start_mode", default="stage1_samples", choices=["stage1_anchor", "stage1_samples", "shared_root_anchor"], help="Initial states used for sample trajectories in direct 2D bifurcation plots.")
    parser.add_argument("--n_stage1_trajectory_samples_per_subtype", type=int, default=20, help="Maximum stage1 sample trajectories per subtype to draw.")
    parser.add_argument("--trajectory_sample_alpha", type=float, default=0.4)
    parser.add_argument("--trajectory_sample_linewidth", type=float, default=1.5)
    parser.add_argument("--mean_trajectory_mode", default="root_to_stage1_anchor_then_model", choices=["root_to_stage1_anchor_then_model", "model_only", "data_anchor"], help="Mean trajectory construction for direct 2D shared-root bifurcation plots.")

    # UMAP/PCA overlay of learned dynamics in the same 2D manifold used by real ROI data.
    parser.add_argument("--plot_dynamics_umap", action="store_true", default=True)
    parser.add_argument("--no_plot_dynamics_umap", dest="plot_dynamics_umap", action="store_false")
    # This is a separate diagnostic view: it fits UMAP/PCA directly on encoded
    # autoencoder latent z, not on the original standardized 68-ROI SUVR space.
    parser.add_argument("--plot_encoded_umap", action="store_true", default=False, help="Fit a separate UMAP/PCA directly on encoded latent z and save diagnostic plots.")
    parser.add_argument("--no_plot_encoded_umap", dest="plot_encoded_umap", action="store_false")
    parser.add_argument("--plot_encoded_dynamics_umap", action="store_true", default=True, help="Overlay learned latent trajectories and velocity arrows on the encoded latent UMAP/PCA view.")
    parser.add_argument("--no_plot_encoded_dynamics_umap", dest="plot_encoded_dynamics_umap", action="store_false")
    parser.add_argument("--trajectory_umap_backend", default="auto", choices=["auto", "umap", "pca"])
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.25)
    parser.add_argument("--umap_metric", default="euclidean")
    parser.add_argument("--umap_max_fit_points", type=int, default=0, help="0 means fit on all displayed real points.")
    parser.add_argument("--encoded_umap_max_fit_points", type=int, default=0, help="0 means fit encoded-latent UMAP/PCA on all displayed encoded points.")
    parser.add_argument("--n_umap_trajectory_samples", type=int, default=60)
    parser.add_argument("--n_encoded_umap_trajectory_samples", type=int, default=60, help="Number of learned latent sample trajectories to draw on the encoded latent UMAP/PCA view.")
    parser.add_argument("--n_encoded_velocity_arrows", type=int, default=80, help="Number of velocity arrows to draw on the encoded latent UMAP/PCA view.")
    parser.add_argument("--encoded_velocity_step", type=float, default=0.3, help="Fallback latent-space step used for encoded velocity arrows when --encoded_velocity_steps is empty.")
    parser.add_argument("--encoded_velocity_steps", default="0.02,0.05,0.10,0.20,0.35", help="Comma-separated latent-space steps for encoded velocity diagnostics.")
    parser.add_argument("--mean_path_mode", default="projected_mean", choices=["projected_mean", "latent_mean"], help="Mean path drawn on encoded trajectory plot; both modes are always saved to CSV.")
    parser.add_argument("--encoded_velocity_arrow_normalize", nargs="?", const=True, default=False, type=parse_bool, help="If true, normalize plotted encoded-UMAP velocity arrows to a fixed display length.")
    parser.add_argument("--encoded_velocity_arrow_length", type=float, default=0.3, help="Display length used when --encoded_velocity_arrow_normalize is true.")
    parser.add_argument("--encoded_velocity_arrow_clip_quantile", type=float, default=0.95, help="Quantile for clipping plotted encoded-UMAP velocity arrow lengths. Set <=0 or >=1 to disable.")

    parser.add_argument("--no_simulate", action="store_true")
    parser.add_argument("--dry_run_ae_only", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_threads", type=int, default=1, help="CPU torch threads. Keep 1 on shared/HPC nodes if runs hang or become very slow.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    set_seed(args.seed)
    if args.num_threads is not None and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    if getattr(args, "pipeline_mode", "train_and_visualize") == "visualize_only":
        run_visualization_only(args=args, out_dir=out_dir, device=device)
        return

    df, roi_cols, prob_cols = prepare_dataframe(args, out_dir)

    autoencoder, z_np, x_hat_raw, rec_mse, mean, std, ae_loss_df, ae_summary = train_autoencoder(
        df=df,
        roi_cols=roi_cols,
        prob_cols=prob_cols,
        args=args,
        device=device,
    )

    ae_loss_df.to_csv(out_dir / "autoencoder_training_loss.csv", index=False)
    save_json(ae_summary, out_dir / "autoencoder_summary.json")

    # Save encoded data and quick reconstruction diagnostics.
    latent_df = df[["_source_row"]].copy()
    for k in range(z_np.shape[1]):
        latent_df[f"z{k}"] = z_np[:, k]
    for col in [args.subtype_col, args.bin_col, args.stage_col, args.use_col, "PTID", "SCANDATE", "Research Group"]:
        if col in df.columns:
            latent_df[col] = df[col].to_numpy()
    latent_df["reconstruction_mse_standardized"] = rec_mse
    latent_df.to_csv(out_dir / "encoded_latent_all_rows.csv", index=False)

    pd.DataFrame({"roi": roi_cols, "mean": mean.squeeze(), "std": std.squeeze()}).to_csv(
        out_dir / "roi_standardization_scaler.csv",
        index=False,
    )

    sep_df = pd.DataFrame()
    encoded_umap_sep_df = pd.DataFrame()
    encoded_projection = None
    if should_visualize(args):
        _, sep_df = visualize_latent(df, z_np, rec_mse, args, out_dir)
        _, encoded_umap_sep_df, encoded_projection = visualize_encoded_latent_umap( # type: ignore
            df,
            z_np,
            rec_mse,
            args,
            out_dir,
            return_projection=True,
        )

    torch.save(
        {
            "autoencoder_state_dict": autoencoder.state_dict(),
            "input_dim": int(autoencoder.encoder[0].in_features),
            "latent_dim": int(autoencoder.bin_head.in_features),
            "ae_hidden_width": int(autoencoder.encoder[0].out_features),
            "n_subtypes": int(autoencoder.subtype_head.out_features),
            "n_bins": int(autoencoder.bin_head.out_features),
            "checkpoint_format_version": 2,
            "roi_cols": roi_cols,
            "roi_mean": mean,
            "roi_std": std,
            "args": vars(args),
        },
        out_dir / "autoencoder_model.pt",
    )

    if args.dry_run_ae_only:
        print(f"AE-only run complete. Outputs saved to: {out_dir}")
        return

    pools = build_latent_pools(
        df=df,
        z_np=z_np,
        prob_cols=prob_cols,
        args=args,
        device=device,
    )
    cfm_model, cfm_loss_df = train_latent_ot_cfm(
        pools=pools,
        n_subtypes=len(prob_cols),
        latent_dim=args.latent_dim,
        args=args,
        device=device,
    )
    cfm_loss_df.to_csv(out_dir / "latent_cfm_training_loss.csv", index=False)

    torch.save(
        {
            "cfm_state_dict": cfm_model.state_dict(),
            "latent_dim": args.latent_dim,
            "n_subtypes": len(prob_cols),
            "cfm_hidden_width": args.cfm_hidden_width,
            "args": vars(args),
        },
        out_dir / "latent_ot_cfm_model.pt",
    )

    config = vars(args).copy()
    config.update(
        {
            "n_rows_after_roi_cleaning": int(len(df)),
            "n_roi_features": int(len(roi_cols)),
            "n_subtypes": int(len(prob_cols)),
            "latent_separation_summary": sep_df.to_dict(orient="records"),
            "encoded_latent_umap_summary": encoded_umap_sep_df.to_dict(orient="records"),
        }
    )
    save_json(config, out_dir / "run_config.json")

    if should_visualize(args) and not args.no_simulate:
        simulate_decode_and_plot(
            autoencoder=autoencoder,
            cfm_model=cfm_model,
            pools=pools,
            df=df,
            roi_cols=roi_cols,
            mean=mean,
            std=std,
            prob_cols=prob_cols,
            args=args,
            device=device,
            z_np=z_np,
            encoded_projection=encoded_projection,
        )

    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['svg.fonttype'] = 'none'
    main()


