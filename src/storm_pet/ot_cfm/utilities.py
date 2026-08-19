#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for conditional OT-based CFM training and analysis."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if sys.platform.startswith("win"):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

try:
    import ot as pot
except ImportError:  # optional
    pot = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # optional
    linear_sum_assignment = None


_OT_PLAN_CACHE: dict[tuple, np.ndarray] = {}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    def to_bool(x):
        if pd.isna(x):
            return False
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (int, float, np.integer, np.floating)):
            return bool(x)
        return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}
    return s.apply(to_bool)


def get_cortical_roi_cols(
    df: pd.DataFrame,
    expected_dim: int | None = 68,
    mode: str = "ctx_lh_rh",
) -> list[str]:
    """Return high-dimensional ROI SUVR columns.

    mode="ctx_lh_rh" selects FreeSurfer cortical hemisphere-specific columns:
        CTX_LH_*_SUVR and CTX_RH_*_SUVR

    This gives 68 cortical ROIs for the uploaded ADNI tau table.
    """
    if mode == "ctx_lh_rh":
        cols = [
            c for c in df.columns
            if c.endswith("_SUVR") and (c.startswith("CTX_LH_") or c.startswith("CTX_RH_"))
        ]
    elif mode == "all_suvr":
        cols = [c for c in df.columns if c.endswith("_SUVR")]
    else:
        raise ValueError(f"Unknown roi_mode={mode!r}")

    cols = sorted(cols)
    if expected_dim is not None and len(cols) != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} ROI columns under roi_mode={mode}, got {len(cols)}. "
            f"Use --expected_dim -1 to disable this check."
        )
    if not cols:
        raise ValueError("No ROI SUVR columns found.")
    return cols


def get_subtype_probability_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.match(r"^subtype_\d+_prob$", c)]
    cols = sorted(cols, key=lambda c: int(re.search(r"^subtype_(\d+)_prob$", c).group(1)))
    if not cols:
        raise ValueError("Cannot find subtype posterior columns like subtype_0_prob.")
    return cols


def fill_missing_by_same_ptid_nearest_date(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    ptid_col: str = "PTID",
    date_col: str = "SCANDATE",
    max_day_diff: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing ROI values from the nearest scan of the same PTID."""
    feature_cols = list(feature_cols)
    out = df.copy()
    if ptid_col not in out.columns or date_col not in out.columns:
        return out, pd.DataFrame()

    out["_SCANDATE_DT_"] = pd.to_datetime(out[date_col], errors="coerce")
    fill_logs = []
    missing_rows = out.index[out[feature_cols].isna().sum(axis=1) > 0].tolist()

    for idx in missing_rows:
        ptid = out.at[idx, ptid_col]
        cur_date = out.at[idx, "_SCANDATE_DT_"]

        for col in feature_cols:
            if not pd.isna(out.at[idx, col]):
                continue

            candidates = out[(out[ptid_col] == ptid) & (out.index != idx) & out[col].notna()][
                ["_SCANDATE_DT_", col]
            ].copy()
            if candidates.empty:
                continue

            if pd.notna(cur_date) and candidates["_SCANDATE_DT_"].notna().any():
                candidates["abs_day_diff"] = (candidates["_SCANDATE_DT_"] - cur_date).abs().dt.days
                candidates = candidates.sort_values(["abs_day_diff", "_SCANDATE_DT_"])
            else:
                candidates["abs_day_diff"] = np.nan

            best = candidates.iloc[0]
            day_diff = best["abs_day_diff"]
            if max_day_diff is not None and pd.notna(day_diff) and day_diff > max_day_diff:
                continue

            out.at[idx, col] = best[col]
            fill_logs.append(
                {
                    "row_index": int(idx),
                    "PTID": ptid,
                    "SCANDATE": out.at[idx, date_col],
                    "feature_col": col,
                    "filled_value": float(best[col]),
                    "source_SCANDATE": best["_SCANDATE_DT_"],
                    "abs_day_diff": day_diff,
                }
            )

    return out.drop(columns=["_SCANDATE_DT_"]), pd.DataFrame(fill_logs)


def numpy_pca_2d(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D PCA scores and explained variance ratio using NumPy SVD."""
    z = np.asarray(z, dtype=np.float64)
    zc = z - z.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(zc, full_matrices=False)
    scores = zc @ vt[:2].T
    var = s ** 2 / max(z.shape[0] - 1, 1)
    evr = var[:2] / max(var.sum(), 1e-12)
    return scores.astype(np.float32), evr.astype(np.float32)


def silhouette_score_np(x: np.ndarray, labels: np.ndarray, max_n: int = 1200, seed: int = 0) -> float:
    """Small dependency-free silhouette score for diagnostics."""
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels)
    valid = pd.notna(labels)
    x = x[valid]
    labels = labels[valid]
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(x) < 3:
        return float("nan")

    if len(x) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_n, replace=False)
        x = x[idx]
        labels = labels[idx]
        uniq = np.unique(labels)

    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)

    vals = []
    for i in range(len(x)):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            continue
        a = d[i, same].mean()
        b = np.inf
        for lab in uniq:
            if lab == labels[i]:
                continue
            mask = labels == lab
            if mask.sum() > 0:
                b = min(b, d[i, mask].mean())
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals)) if vals else float("nan")


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def numeric_series(values, index: pd.Index | None = None) -> pd.Series:
    """Convert an array-like object to a numeric Series with NaN on failure."""
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def add_stage0_extra_dynamics_bin(df: pd.DataFrame, args, out_dir: Path) -> pd.DataFrame:
    """Create a common dynamics bin column with stage 0 as an explicit extra bin.

    Typical input from the common-bin script has:
        coarse_stage_bin = 0,1,2,3 for positive-stage bins only,
        sustain_stage = 0 for stage-0 reference rows.

    This function creates, by default:
        dynamics_bin = 0 for sustain_stage == 0,
        dynamics_bin = coarse_stage_bin + 1 for positive-stage rows.

    Therefore the learned time axis becomes:
        0: S0, 1: first positive-stage bin, ..., K: last positive-stage bin.
    """
    if not getattr(args, "include_stage0_as_bin", False):
        if args.bin_col in df.columns:
            df[args.bin_col] = numeric_series(df[args.bin_col])
        return df

    if args.stage_col not in df.columns:
        raise ValueError(f"--include_stage0_as_bin requires stage column {args.stage_col!r}.")

    target_col = args.bin_col
    source_col = args.source_bin_col
    stage = numeric_series(df[args.stage_col])

    if source_col in df.columns:
        source_bin = numeric_series(df[source_col])
    else:
        source_bin = pd.Series(np.nan, index=df.index, dtype=float)

    # Preserve the original positive-stage adaptive bin if the target column overwrites it.
    if source_col in df.columns:
        preserved_col = f"{source_col}_original"
        if preserved_col not in df.columns:
            df[preserved_col] = source_bin

    dynamics_bin = pd.Series(np.nan, index=df.index, dtype=float)
    stage0_mask = stage == args.stage0_raw_value
    positive_mask = source_bin.notna() & ~stage0_mask

    dynamics_bin.loc[stage0_mask] = int(args.stage0_bin_value)
    dynamics_bin.loc[positive_mask] = source_bin.loc[positive_mask].astype(int) + int(args.positive_bin_shift)
    df[target_col] = dynamics_bin

    # Label bins using observed raw SuStaIn stage ranges.
    label_col = args.dynamics_label_col
    df[label_col] = pd.NA
    df.loc[stage0_mask, label_col] = args.stage0_label

    for b in sorted(df.loc[df[target_col].notna(), target_col].astype(int).unique().tolist()):
        m = df[target_col].astype("Int64") == b
        if b == int(args.stage0_bin_value):
            df.loc[m, label_col] = args.stage0_label
            continue
        stage_vals = stage.loc[m].dropna().astype(int)
        if stage_vals.empty:
            label = f"bin{b}"
        else:
            a = int(stage_vals.min())
            c = int(stage_vals.max())
            label = f"S{a}" if a == c else f"S{a}-{c}"
        df.loc[m, label_col] = label

    # Make the dynamics inclusion flag consistent with the new bin column.
    if args.use_col:
        df[args.use_col] = df[target_col].notna()

    bin_summary = (
        df[df[target_col].notna()]
        .groupby([target_col, label_col], dropna=False)
        .size()
        .reset_index(name="n_samples")
        .sort_values(target_col)
    )
    bin_summary.to_csv(out_dir / "stage0_extra_dynamics_bin_summary.csv", index=False)

    if args.subtype_col in df.columns:
        subtype_summary = (
            df[df[target_col].notna()]
            .groupby([args.subtype_col, target_col, label_col], dropna=False)
            .size()
            .reset_index(name="n_samples")
            .sort_values([args.subtype_col, target_col])
        )
        subtype_summary.to_csv(out_dir / "stage0_extra_dynamics_bin_count_by_subtype.csv", index=False)

    return df


def compute_scaler(x: np.ndarray, eps: float = 1e-6):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass
class LatentPool:
    subtype: int | str
    bin_index: int
    z: torch.Tensor
    weights: torch.Tensor
    row_indices: np.ndarray
    effective_n: float
    mean_posterior: float
    n: int


@dataclass
class LatentTransition:
    source: LatentPool
    target: LatentPool
    cond_subtype: int
    source_subtype: int | str
    target_subtype: int
    source_bin: int
    target_bin: int
    transition_type: str
    is_shared_root: bool
    loss_weight: float = 1.0


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return float(1.0 / np.sum(weights * weights))


def _is_shared_root_key(key) -> bool:
    return isinstance(key, tuple) and len(key) == 2 and key[0] == "shared"


def _transition_to_row(transition: LatentTransition) -> dict:
    return {
        "is_shared_root": bool(transition.is_shared_root),
        "source_bin": int(transition.source_bin),
        "target_bin": int(transition.target_bin),
        "source_subtype": transition.source_subtype,
        "target_subtype": transition.target_subtype,
        "condition_subtype": int(transition.cond_subtype),
        "transition_type": transition.transition_type,
        "source_n": int(transition.source.n),
        "target_n": int(transition.target.n),
        "source_effective_n": float(transition.source.effective_n),
        "target_effective_n": float(transition.target.effective_n),
        "loss_weight": float(transition.loss_weight),
    }


def build_transitions(pools: dict[tuple[int | str, int], LatentPool], args=None):
    if args is None:
        transitions = []
        for subtype in sorted({s for s, _ in pools if s != "shared"}):
            bins = sorted(b for s, b in pools if s == subtype)
            for b0, b1 in zip(bins[:-1], bins[1:]):
                transitions.append((pools[(subtype, b0)], pools[(subtype, b1)]))
        if not transitions:
            raise ValueError("No adjacent latent-bin transitions were found.")
        return transitions

    shared_root_stage0 = bool(getattr(args, "shared_root_stage0", False))
    branch_loss_weight = float(getattr(args, "branch_loss_weight", 1.0))
    transitions = []

    if shared_root_stage0:
        root_bin = int(getattr(args, "root_bin_value", 0))
        start_bin = int(getattr(args, "subtype_specific_start_bin", root_bin + 1))
        root_key = ("shared", root_bin)
        if root_key not in pools:
            raise ValueError(f"Shared root pool {root_key!r} was not built.")

        branch_subtypes = sorted(
            int(s)
            for s, b in pools
            if s != "shared" and int(b) >= start_bin
        )
        branch_subtypes = sorted(set(branch_subtypes))
        root_loss_each = float(getattr(args, "root_loss_weight", 1.0)) / max(len(branch_subtypes), 1)

        for subtype in branch_subtypes:
            bins = sorted(int(b) for s, b in pools if s == subtype and int(b) >= start_bin)
            if not bins:
                print(f"WARNING: subtype {subtype} has no non-root bins; skipping shared-root transition.")
                continue

            first_bin = bins[0]
            transitions.append(
                LatentTransition(
                    source=pools[root_key],
                    target=pools[(subtype, first_bin)],
                    cond_subtype=subtype,
                    source_subtype="shared",
                    target_subtype=subtype,
                    source_bin=root_bin,
                    target_bin=first_bin,
                    transition_type="shared_root_to_subtype",
                    is_shared_root=True,
                    loss_weight=root_loss_each,
                )
            )

            for b0, b1 in zip(bins[:-1], bins[1:]):
                transitions.append(
                    LatentTransition(
                        source=pools[(subtype, b0)],
                        target=pools[(subtype, b1)],
                        cond_subtype=subtype,
                        source_subtype=subtype,
                        target_subtype=subtype,
                        source_bin=int(b0),
                        target_bin=int(b1),
                        transition_type="within_subtype",
                        is_shared_root=False,
                        loss_weight=branch_loss_weight,
                    )
                )
    else:
        for subtype in sorted({s for s, _ in pools if s != "shared"}):
            bins = sorted(int(b) for s, b in pools if s == subtype)
            for b0, b1 in zip(bins[:-1], bins[1:]):
                transitions.append(
                    LatentTransition(
                        source=pools[(subtype, b0)],
                        target=pools[(subtype, b1)],
                        cond_subtype=int(subtype),
                        source_subtype=int(subtype),
                        target_subtype=int(subtype),
                        source_bin=int(b0),
                        target_bin=int(b1),
                        transition_type="within_subtype",
                        is_shared_root=False,
                        loss_weight=branch_loss_weight,
                    )
                )

    if not transitions:
        raise ValueError("No adjacent latent-bin transitions were found.")

    out_dir = getattr(args, "out_dir", None)
    if out_dir:
        pd.DataFrame([_transition_to_row(t) for t in transitions]).to_csv(
            Path(out_dir) / "latent_transition_summary.csv",
            index=False,
        )
    return transitions


def sample_pool(pool: LatentPool, batch_size: int) -> torch.Tensor:
    idx = torch.multinomial(pool.weights, num_samples=batch_size, replacement=True)
    return pool.z[idx]


def _normalize_ot_weights(weights, n: int) -> np.ndarray:
    if weights is None:
        arr = np.ones(n, dtype=np.float64)
    elif torch.is_tensor(weights):
        arr = weights.detach().cpu().numpy().astype(np.float64)
    else:
        arr = np.asarray(weights, dtype=np.float64)

    arr = arr.reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"OT weights length {arr.shape[0]} does not match samples {n}.")

    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    total = float(arr.sum())
    if total <= 1e-12:
        arr = np.ones(n, dtype=np.float64)
        total = float(n)
    return arr / total


def _logsumexp_np(x: np.ndarray, axis: int) -> np.ndarray:
    max_x = np.max(x, axis=axis, keepdims=True)
    max_x = np.where(np.isfinite(max_x), max_x, 0.0)
    summed = np.sum(np.exp(x - max_x), axis=axis, keepdims=True)
    out = np.log(np.maximum(summed, 1e-300)) + max_x
    return np.squeeze(out, axis=axis)


def _sinkhorn_balanced_plan(a: np.ndarray, b: np.ndarray, cost: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Entropic balanced OT fallback used when POT is unavailable."""
    cost = np.asarray(cost, dtype=np.float64)
    finite_cost = cost[np.isfinite(cost)]
    if finite_cost.size == 0:
        return np.outer(a, b)

    max_finite = float(np.max(finite_cost))
    cost = np.where(np.isfinite(cost), cost, max_finite)
    positive_cost = finite_cost[finite_cost > 0.0]
    if positive_cost.size:
        cost_scale = float(np.median(positive_cost))
    else:
        cost_scale = max(float(np.ptp(finite_cost)), 1.0)
    reg = max(0.05 * cost_scale, 1e-3)

    eps = 1e-12
    a_safe = np.maximum(a.astype(np.float64), eps)
    b_safe = np.maximum(b.astype(np.float64), eps)
    a_safe = a_safe / a_safe.sum()
    b_safe = b_safe / b_safe.sum()

    log_k = -cost / reg
    log_a = np.log(a_safe)
    log_b = np.log(b_safe)
    log_u = np.zeros_like(log_a)
    log_v = np.zeros_like(log_b)

    for _ in range(max_iter):
        log_u = log_a - _logsumexp_np(log_k + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp_np(log_k.T + log_u[None, :], axis=1)

    log_pi = log_k + log_u[:, None] + log_v[None, :]
    return np.exp(log_pi)


def _ot_plan_cache_key(
    z0: torch.Tensor,
    z1: torch.Tensor,
    w0,
    w1,
) -> tuple | None:
    parts = [id(pot)]
    for tensor in (z0, z1, w0, w1):
        if tensor is None:
            parts.append(None)
            continue
        if not torch.is_tensor(tensor):
            return None
        parts.append((tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype), str(tensor.device)))
    return tuple(parts)


def _balanced_ot_plan(
    z0: torch.Tensor,
    z1: torch.Tensor,
    w0=None,
    w1=None,
    cost: torch.Tensor | None = None,
) -> np.ndarray:
    key = _ot_plan_cache_key(z0, z1, w0, w1)
    if key is not None and key in _OT_PLAN_CACHE:
        return _OT_PLAN_CACHE[key]

    if cost is None:
        with torch.no_grad():
            cost = torch.cdist(z0, z1) ** 2

    a = _normalize_ot_weights(w0, z0.shape[0])
    b = _normalize_ot_weights(w1, z1.shape[0])
    cost_np = cost.detach().cpu().numpy().astype(np.float64)

    if pot is not None:
        pi = pot.emd(a, b, cost_np)
    else:
        pi = _sinkhorn_balanced_plan(a, b, cost_np)

    if (not np.all(np.isfinite(pi))) or abs(pi.sum()) < 1e-12:
        pi = np.outer(a, b)
    else:
        pi = np.maximum(pi, 0.0)
        pi = pi / max(float(pi.sum()), 1e-12)

    if key is not None:
        _OT_PLAN_CACHE[key] = pi
    return pi


def sample_ot_pairs(
    z0: torch.Tensor,
    z1: torch.Tensor,
    w0=None,
    w1=None,
    num_pairs: int | None = None,
):
    if num_pairs is None:
        num_pairs = z0.shape[0]

    if w0 is None and w1 is None and pot is None and z0.shape[0] == z1.shape[0] and num_pairs == z0.shape[0]:
        with torch.no_grad():
            cost = torch.cdist(z0, z1) ** 2
        if linear_sum_assignment is None:
            perm = torch.randperm(num_pairs, device=z0.device)
            return z0, z1[perm]
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        return z0[torch.as_tensor(row_ind, device=z0.device)], z1[torch.as_tensor(col_ind, device=z0.device)]

    pi = _balanced_ot_plan(z0, z1, w0=w0, w1=w1)
    choices = np.random.choice(pi.size, size=num_pairs, replace=True, p=pi.reshape(-1))
    i, j = np.divmod(choices, pi.shape[1])
    return z0[torch.as_tensor(i, device=z0.device)], z1[torch.as_tensor(j, device=z0.device)]


def one_hot_subtype(subtype: int, n_subtypes: int, batch_size: int, device: torch.device) -> torch.Tensor:
    cond = torch.zeros(batch_size, n_subtypes, device=device)
    cond[:, subtype] = 1.0
    return cond


def _transition_parts(transition):
    if isinstance(transition, LatentTransition):
        return transition.source, transition.target, transition.cond_subtype, transition.loss_weight
    src, tgt = transition
    return src, tgt, int(src.subtype), 1.0


def sample_cfm_batch(
    transitions,
    n_subtypes: int,
    batch_size: int,
    sigma: float,
    device: torch.device,
    return_loss_weights: bool = False,
):
    ts, zts, uts, conds, weights = [], [], [], [], []
    for transition in transitions:
        src, tgt, cond_subtype, loss_weight = _transition_parts(transition)
        z0, z1 = sample_ot_pairs(src.z, tgt.z, src.weights, tgt.weights, num_pairs=batch_size)

        local_t = torch.rand(batch_size, device=device)
        t_pad = local_t[:, None]
        eps = torch.randn_like(z0)
        zt = (1.0 - t_pad) * z0 + t_pad * z1 + sigma * eps
        ut = z1 - z0
        global_t = local_t + float(src.bin_index)
        cond = one_hot_subtype(int(cond_subtype), n_subtypes, batch_size, device)

        ts.append(global_t)
        zts.append(zt)
        uts.append(ut)
        conds.append(cond)
        weights.append(torch.full((batch_size,), float(loss_weight), dtype=torch.float32, device=device))

    out = (torch.cat(ts), torch.cat(zts), torch.cat(uts), torch.cat(conds))
    if return_loss_weights:
        return (*out, torch.cat(weights))
    return out


def euler_trajectory_latent(
    model,
    z0: torch.Tensor,
    subtype: int,
    n_subtypes: int,
    t_eval: torch.Tensor,
    device: torch.device,
    latent_velocity_clip: float = 0.0,
):
    model.eval()
    zs = [z0.detach().clone()]
    z = z0.detach().clone()
    cond = one_hot_subtype(subtype, n_subtypes, z.shape[0], device)

    with torch.no_grad():
        for i in range(len(t_eval) - 1):
            t0 = t_eval[i]
            dt = t_eval[i + 1] - t0
            t_batch = t0.repeat(z.shape[0])
            v = model(z, t_batch, cond)
            if latent_velocity_clip > 0:
                norm = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
                scale = torch.clamp(float(latent_velocity_clip) / norm, max=1.0)
                v = v * scale
            z = z + dt * v
            zs.append(z.detach().clone())
    return torch.stack(zs, dim=0)


def decode_trajectory(autoencoder, z_traj: torch.Tensor, mean: np.ndarray, std: np.ndarray, device: torch.device, clip_low: np.ndarray | None = None, clip_high: np.ndarray | None = None):
    autoencoder.eval()
    T, N, D = z_traj.shape
    with torch.no_grad():
        z_flat = z_traj.reshape(T * N, D)
        x_std = autoencoder.decode(z_flat).reshape(T, N, -1).detach().cpu().numpy()
    x_raw = x_std * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    if clip_low is not None and clip_high is not None:
        x_raw = np.clip(x_raw, clip_low.reshape(1, 1, -1), clip_high.reshape(1, 1, -1))
    return x_raw.astype(np.float32)


