from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from storm_pet.exceptions import ArtifactValidationError


@dataclass(frozen=True)
class SustainPreprocessor:
    """Frozen control normalization and event definition used by a SuStaIn model."""

    input_features: tuple[str, ...]
    feature_mask: np.ndarray
    final_features: tuple[str, ...]
    control_mean: np.ndarray
    control_std: np.ndarray
    z_vals: np.ndarray
    z_max: np.ndarray
    clip_min: float
    clip_max: float
    ddof: int = 0

    def __post_init__(self) -> None:
        n_input = len(self.input_features)
        n_final = len(self.final_features)
        if self.feature_mask.shape != (n_input,):
            raise ArtifactValidationError("feature_mask must contain one value per input feature")
        if self.feature_mask.dtype.kind != "b":
            raise ArtifactValidationError("feature_mask must be boolean")
        if int(self.feature_mask.sum()) != n_final:
            raise ArtifactValidationError("feature_mask and final_features disagree")
        if self.control_mean.shape != (n_input,) or self.control_std.shape != (n_input,):
            raise ArtifactValidationError("control statistics must match input_features")
        if np.any(~np.isfinite(self.control_mean)) or np.any(~np.isfinite(self.control_std)):
            raise ArtifactValidationError("control statistics contain NaN or infinity")
        if np.any(self.control_std <= 0):
            raise ArtifactValidationError("control standard deviations must be positive")
        if self.z_vals.ndim != 2 or self.z_vals.shape[0] != n_final:
            raise ArtifactValidationError("z_vals must have one row per final feature")
        if self.z_max.shape != (n_final,):
            raise ArtifactValidationError("z_max must have one value per final feature")
        if np.any(self.z_vals < 0) or np.any(self.z_max <= 0):
            raise ArtifactValidationError("event thresholds and maxima must be non-negative")
        if np.any(np.max(self.z_vals, axis=1) > self.z_max):
            raise ArtifactValidationError("z_max cannot be below a retained event threshold")
        if self.clip_min >= self.clip_max:
            raise ArtifactValidationError("clip_min must be below clip_max")

    @property
    def stage_max(self) -> int:
        return int(np.count_nonzero(self.z_vals))

    def transform(self, values: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        names = tuple(str(name) for name in feature_names)
        if names != self.input_features:
            raise ArtifactValidationError(
                "new-data feature names/order differ from the fitted SuStaIn contract"
            )
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(names):
            raise ArtifactValidationError("new data must be a 2D scan-by-feature matrix")
        if np.any(~np.isfinite(array)):
            raise ArtifactValidationError("new SuStaIn input contains NaN or infinity")
        z = (array - self.control_mean) / self.control_std
        return np.clip(z[:, self.feature_mask], self.clip_min, self.clip_max)

    def metadata(self) -> dict[str, Any]:
        return {
            "input_features": list(self.input_features),
            "final_features": list(self.final_features),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "ddof": self.ddof,
            "stage_max": self.stage_max,
        }


def _zmax_for(last_event: float) -> float:
    # Preserve the convention used by the original repository: low terminal events
    # extrapolate to 2/3, while all larger event grids terminate at z=5.
    if last_event <= 1:
        return 2.0
    if last_event <= 2:
        return 3.0
    return 5.0


def fit_sustain_preprocessor(
    values: np.ndarray,
    control_mask: np.ndarray,
    disease_mask: np.ndarray,
    feature_names: Sequence[str],
    event_zscores: Sequence[float],
    *,
    minimum_supporting_scans: int = 10,
    drop_non_increasing_disease_mean: bool = True,
    ddof: int = 0,
    clip_min: float = -3.0,
    clip_max: float = 10.0,
    terminal_zmax: float | None = None,
) -> SustainPreprocessor:
    """Fit only the preprocessing definition; this function never trains SuStaIn."""

    x = np.asarray(values, dtype=float)
    controls = np.asarray(control_mask, dtype=bool)
    disease = np.asarray(disease_mask, dtype=bool)
    names = tuple(str(name) for name in feature_names)
    thresholds = tuple(float(z) for z in event_zscores)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError("values must be a scan-by-feature matrix matching feature_names")
    if controls.shape != (x.shape[0],) or disease.shape != (x.shape[0],):
        raise ValueError("control_mask and disease_mask must contain one value per scan")
    if not controls.any() or not disease.any():
        raise ValueError("both control and disease groups must be non-empty")
    if np.any(~np.isfinite(x)):
        raise ValueError("values contain NaN or infinity")
    if not thresholds or any(z <= 0 for z in thresholds):
        raise ValueError("event_zscores must contain positive thresholds")
    if tuple(sorted(thresholds)) != thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("event_zscores must be unique and increasing")
    if minimum_supporting_scans < 1:
        raise ValueError("minimum_supporting_scans must be positive")
    if terminal_zmax is not None and terminal_zmax < max(thresholds):
        raise ValueError("terminal_zmax cannot be below the largest event threshold")

    mean = x[controls].mean(axis=0)
    std = x[controls].std(axis=0, ddof=ddof)
    std = np.where(std == 0, 1e-6, std)
    z_all = np.clip((x - mean) / std, clip_min, clip_max)
    increasing = np.ones(x.shape[1], dtype=bool)
    if drop_non_increasing_disease_mean:
        increasing = z_all[disease].mean(axis=0) > z_all[controls].mean(axis=0)

    keep = np.zeros(x.shape[1], dtype=bool)
    rows: list[list[float]] = []
    zmax: list[float] = []
    for index in np.flatnonzero(increasing):
        supported = [z for z in thresholds if int(np.count_nonzero(z_all[:, index] > z)) >= minimum_supporting_scans]
        if not supported:
            continue
        keep[index] = True
        # pySuStaIn treats zero entries as absent events. Left-aligning is the
        # behavior of the legacy scripts and remains unambiguous after export.
        rows.append(supported + [0.0] * (len(thresholds) - len(supported)))
        zmax.append(
            float(terminal_zmax)
            if terminal_zmax is not None
            else _zmax_for(max(supported))
        )

    if not keep.any():
        raise ValueError("no feature has an increasing disease mean and a supported event")
    final_names = tuple(name for name, selected in zip(names, keep, strict=True) if selected)
    return SustainPreprocessor(
        input_features=names,
        feature_mask=keep,
        final_features=final_names,
        control_mean=mean,
        control_std=std,
        z_vals=np.asarray(rows, dtype=float),
        z_max=np.asarray(zmax, dtype=float),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
        ddof=int(ddof),
    )
