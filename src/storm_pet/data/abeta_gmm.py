from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from storm_pet.exceptions import ConfigurationError


@dataclass(frozen=True)
class AbetaGmmFit:
    labels: pd.Series
    positive_probability: pd.Series
    means: tuple[float, float]
    standard_deviations: tuple[float, float]
    weights: tuple[float, float]
    negative_component: int
    positive_component: int
    posterior_cutoff: float
    legacy_grid_cutoff: float


def extract_roi_suvr_columns(
    data: pd.DataFrame, expected_count: int = 163
) -> list[str]:
    columns = [column for column in data.columns if column.endswith("_SUVR")]
    roi_columns = [column for column in columns if column != "SUMMARY_SUVR"]
    if len(roi_columns) != expected_count:
        raise ConfigurationError(
            f"Expected {expected_count} ROI SUVR columns after excluding SUMMARY_SUVR, "
            f"found {len(roi_columns)}"
        )
    return roi_columns


def _weighted_component_density(
    model: GaussianMixture, grid: np.ndarray
) -> np.ndarray:
    log_probability = model.score_samples(grid)
    responsibilities = model.predict_proba(grid)
    return np.exp(log_probability)[:, None] * responsibilities


def _intersection_cutoff(
    model: GaussianMixture, means: np.ndarray, restrict_between_means: bool
) -> float:
    lower = float(means.min()) if restrict_between_means else float(model.means_.min())
    upper = float(means.max()) if restrict_between_means else float(model.means_.max())
    if not restrict_between_means:
        # Replaced by the caller's observed range for exact legacy reproduction.
        raise AssertionError("Full-range cutoff requires an explicit observed range")
    grid = np.linspace(lower, upper, 10000, dtype=float).reshape(-1, 1)
    density = _weighted_component_density(model, grid)
    index = int(np.argmin(np.abs(density[:, 0] - density[:, 1])))
    return float(grid[index, 0])


def fit_abeta_gmm(
    data: pd.DataFrame,
    summary_column: str = "SUMMARY_SUVR",
    random_seed: int = 42,
    n_init: int = 20,
) -> AbetaGmmFit:
    if summary_column not in data.columns:
        raise ConfigurationError(f"Missing Aβ summary column: {summary_column}")
    values = pd.to_numeric(data[summary_column], errors="coerce")
    valid = values.notna()
    if valid.sum() < 2:
        raise ConfigurationError("At least two valid SUMMARY_SUVR values are required")
    matrix = values.loc[valid].to_numpy(np.float64).reshape(-1, 1)
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=random_seed,
        n_init=n_init,
    ).fit(matrix)

    means = model.means_.reshape(-1)
    deviations = np.sqrt(model.covariances_.reshape(-1))
    weights = model.weights_.reshape(-1)
    negative_component = int(np.argmin(means))
    positive_component = int(np.argmax(means))
    probabilities = model.predict_proba(matrix)[:, positive_component]

    label_series = pd.Series(pd.NA, index=data.index, dtype="Int64")
    probability_series = pd.Series(np.nan, index=data.index, dtype=float)
    label_series.loc[valid] = (probabilities > 0.5).astype(int)
    probability_series.loc[valid] = probabilities

    posterior_cutoff = _intersection_cutoff(model, means, restrict_between_means=True)
    legacy_grid = np.linspace(matrix.min(), matrix.max(), 5000).reshape(-1, 1)
    legacy_density = _weighted_component_density(model, legacy_grid)
    legacy_index = int(np.argmin(np.abs(legacy_density[:, 0] - legacy_density[:, 1])))

    return AbetaGmmFit(
        labels=label_series,
        positive_probability=probability_series,
        means=tuple(float(value) for value in means),
        standard_deviations=tuple(float(value) for value in deviations),
        weights=tuple(float(value) for value in weights),
        negative_component=negative_component,
        positive_component=positive_component,
        posterior_cutoff=posterior_cutoff,
        legacy_grid_cutoff=float(legacy_grid[legacy_index, 0]),
    )


def assign_abeta_status(data: pd.DataFrame, fit: AbetaGmmFit) -> pd.DataFrame:
    output = data.copy()
    output["ABETA_GMM_PROB"] = fit.positive_probability
    output["ABETA_GMM_LABEL"] = fit.labels
    output["ABETA_GMM_CUTOFF_SUMMARY_SUVR"] = fit.posterior_cutoff
    output["ABETA_GMM_LEGACY_GRID_CUTOFF"] = fit.legacy_grid_cutoff
    return output

