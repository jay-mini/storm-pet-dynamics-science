from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd

from storm_pet.config import load_yaml
from storm_pet.data.features import aggregate_volume_weighted_features, load_feature_map
from storm_pet.data.roi_schema import suvr_to_volume_column
from storm_pet.exceptions import ArtifactValidationError, ConfigurationError
from storm_pet.paths import repository_root
from storm_pet.sustain.artifacts import SustainBundle, load_sustain_bundle
from storm_pet.sustain.inference import SustainInferenceResult
from storm_pet.sustain.zscore_backend import PreparedZScoreSustainBackend

if TYPE_CHECKING:
    from storm_pet.sustain.registry import SustainModelRegistry


@dataclass(frozen=True)
class SustainPrediction:
    scan_id: str
    subtype: int
    stage: int
    subtype_probability: float
    stage_probability: float
    subtype_probabilities: tuple[float, ...]
    stage_probabilities: tuple[float, ...]


class SustainPredictor:
    """Read-only, precompiled SuStaIn predictor for interactive requests."""

    def __init__(
        self,
        *,
        bundle: SustainBundle,
        feature_map: Mapping[str, Sequence[str]],
        tracer: str,
        reference_region: str,
        posterior_samples: int = 1000,
        max_batch_size: int = 32,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        normalized_map = {
            str(feature): [str(column) for column in columns]
            for feature, columns in feature_map.items()
        }
        if tuple(normalized_map) != bundle.preprocessor.input_features:
            raise ArtifactValidationError(
                "feature-map order differs from the frozen SuStaIn input contract"
            )
        self.bundle = bundle
        self.feature_map = normalized_map
        self.tracer = tracer
        self.reference_region = reference_region
        self.max_batch_size = max_batch_size
        self.backend = PreparedZScoreSustainBackend.from_bundle(
            bundle, n_samples=posterior_samples
        )
        suvr_columns = [column for columns in normalized_map.values() for column in columns]
        self.required_columns = tuple(
            dict.fromkeys(suvr_columns + [suvr_to_volume_column(column) for column in suvr_columns])
        )

    @classmethod
    def from_registry(
        cls,
        model_id: str,
        *,
        registry: SustainModelRegistry | None = None,
        posterior_samples: int = 1000,
        max_batch_size: int = 32,
    ) -> "SustainPredictor":
        from storm_pet.sustain.registry import SustainModelRegistry

        source = registry or SustainModelRegistry()
        registration = source.get(model_id)
        bundle = load_sustain_bundle(registration.bundle_path)
        config = load_yaml(registration.config_path)
        if bundle.model_id != model_id or bundle.modality != registration.modality:
            raise ArtifactValidationError("registry and SuStaIn bundle identity disagree")
        if bundle.selected_subtypes != registration.expected_subtypes:
            raise ArtifactValidationError("registry and SuStaIn subtype counts disagree")
        if bundle.preprocessor.stage_max != registration.expected_stage_max:
            raise ArtifactValidationError("registry and SuStaIn stage counts disagree")
        feature_map_path = repository_root() / str(config["feature_map"])
        return cls(
            bundle=bundle,
            feature_map=load_feature_map(feature_map_path),
            tracer=str(config["tracer"]),
            reference_region=str(config["reference_region"]),
            posterior_samples=posterior_samples,
            max_batch_size=max_batch_size,
        )

    def schema(self) -> dict[str, object]:
        return {
            "model_id": self.bundle.model_id,
            "modality": self.bundle.modality,
            "tracer": self.tracer,
            "reference_region": self.reference_region,
            "required_columns": list(self.required_columns),
            "regional_features": list(self.bundle.preprocessor.input_features),
            "selected_subtypes": self.bundle.selected_subtypes,
            "stage_min": 0,
            "stage_max": self.bundle.preprocessor.stage_max,
            "max_batch_size": self.max_batch_size,
        }

    def predict_records(
        self,
        records: Sequence[tuple[str, Mapping[str, float]]],
    ) -> list[SustainPrediction]:
        if not records:
            raise ConfigurationError("at least one scan is required")
        if len(records) > self.max_batch_size:
            raise ConfigurationError(
                f"at most {self.max_batch_size} scans may be inferred per request"
            )
        scan_ids = [str(scan_id).strip() for scan_id, _values in records]
        if any(not scan_id for scan_id in scan_ids) or len(set(scan_ids)) != len(scan_ids):
            raise ConfigurationError("scan_id values must be non-empty and unique")
        frame = pd.DataFrame([dict(values) for _scan_id, values in records])
        missing = sorted(set(self.required_columns).difference(frame.columns))
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ConfigurationError(f"missing required ROI columns: {preview}{suffix}")
        aggregated = aggregate_volume_weighted_features(frame, self.feature_map).values
        if aggregated.isna().any(axis=None):
            raise ConfigurationError("a scan has no valid positive-volume ROI for a regional feature")
        z_data = self.bundle.preprocessor.transform(
            aggregated.to_numpy(float), self.bundle.preprocessor.input_features
        )
        outputs = self.backend.subtype_and_stage_individuals_newData(z_data)
        result = SustainInferenceResult(*(np.asarray(value) for value in outputs))
        self._validate_result(result, len(records))
        predictions = []
        for index, scan_id in enumerate(scan_ids):
            predictions.append(
                SustainPrediction(
                    scan_id=scan_id,
                    subtype=int(result.ml_subtype[index, 0]),
                    stage=int(result.ml_stage[index, 0]),
                    subtype_probability=float(result.prob_ml_subtype[index, 0]),
                    stage_probability=float(result.prob_ml_stage[index, 0]),
                    subtype_probabilities=tuple(float(x) for x in result.prob_subtype[index]),
                    stage_probabilities=tuple(float(x) for x in result.prob_stage[index]),
                )
            )
        return predictions

    def _validate_result(self, result: SustainInferenceResult, n_scans: int) -> None:
        arrays = [
            result.ml_subtype,
            result.prob_ml_subtype,
            result.ml_stage,
            result.prob_ml_stage,
            result.prob_subtype,
            result.prob_stage,
            result.prob_subtype_stage,
        ]
        if any(array.shape[0] != n_scans or not np.isfinite(array).all() for array in arrays):
            raise ArtifactValidationError("SuStaIn returned invalid or non-finite outputs")
        if not np.allclose(result.prob_subtype.sum(axis=1), 1.0, atol=1e-8):
            raise ArtifactValidationError("SuStaIn subtype probabilities are not normalized")
        if not np.allclose(result.prob_stage.sum(axis=1), 1.0, atol=1e-8):
            raise ArtifactValidationError("SuStaIn stage probabilities are not normalized")
        if np.any(result.ml_subtype < 0) or np.any(
            result.ml_subtype >= self.bundle.selected_subtypes
        ):
            raise ArtifactValidationError("SuStaIn returned an invalid subtype")
        if np.any(result.ml_stage < 0) or np.any(
            result.ml_stage > self.bundle.preprocessor.stage_max
        ):
            raise ArtifactValidationError("SuStaIn returned an invalid stage")
