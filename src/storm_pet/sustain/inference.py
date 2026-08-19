from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from storm_pet.sustain.artifacts import SustainBundle


class NewDataInferenceBackend(Protocol):
    def subtype_and_stage_individuals_newData(
        self,
        data_new: np.ndarray,
        samples_sequence: np.ndarray,
        samples_f: np.ndarray,
        n_samples: int,
    ) -> tuple[np.ndarray, ...]: ...


@dataclass(frozen=True)
class SustainInferenceResult:
    ml_subtype: np.ndarray
    prob_ml_subtype: np.ndarray
    ml_stage: np.ndarray
    prob_ml_stage: np.ndarray
    prob_subtype: np.ndarray
    prob_stage: np.ndarray
    prob_subtype_stage: np.ndarray


def infer_new_data(
    bundle: SustainBundle,
    raw_features: np.ndarray,
    feature_names: Sequence[str],
    backend: NewDataInferenceBackend,
    *,
    posterior_samples: int = 1000,
) -> SustainInferenceResult:
    """Transform new scans with frozen training statistics and invoke pySuStaIn-compatible inference."""

    bundle.validate()
    if posterior_samples < 1:
        raise ValueError("posterior_samples must be positive")
    z_data = bundle.preprocessor.transform(raw_features, feature_names)
    outputs = backend.subtype_and_stage_individuals_newData(
        z_data,
        bundle.samples_sequence,
        bundle.samples_f,
        min(posterior_samples, bundle.samples_sequence.shape[2]),
    )
    if len(outputs) != 7:
        raise ValueError("SuStaIn inference backend must return seven output arrays")
    result = SustainInferenceResult(*(np.asarray(value) for value in outputs))
    n_scans = z_data.shape[0]
    if result.ml_subtype.shape[0] != n_scans or result.ml_stage.shape[0] != n_scans:
        raise ValueError("SuStaIn backend returned a different number of scans")
    return result
