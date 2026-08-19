from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from storm_pet.exceptions import ConfigurationError
from storm_pet.data.roi_schema import suvr_to_volume_column


@dataclass(frozen=True)
class FeatureAggregationResult:
    values: pd.DataFrame
    valid_roi_counts: pd.DataFrame
    total_volumes: pd.DataFrame


def load_feature_map(path: Path) -> dict[str, list[str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), dict):
        raise ConfigurationError(f"Feature map must contain a features mapping: {path}")
    result: dict[str, list[str]] = {}
    for feature, columns in payload["features"].items():
        if not isinstance(feature, str) or not isinstance(columns, list) or not columns:
            raise ConfigurationError(f"Invalid feature definition for {feature!r}: {path}")
        normalized = [str(column) for column in columns]
        if len(normalized) != len(set(normalized)):
            raise ConfigurationError(f"Feature {feature} contains duplicate ROI columns")
        result[feature] = normalized
    return result


def aggregate_volume_weighted_features(
    data: pd.DataFrame,
    feature_map: dict[str, list[str]],
    minimum_valid_rois: int = 1,
) -> FeatureAggregationResult:
    if minimum_valid_rois < 1:
        raise ConfigurationError("minimum_valid_rois must be at least one")

    required_suvr = [column for columns in feature_map.values() for column in columns]
    required_volume = [suvr_to_volume_column(column) for column in required_suvr]
    missing = sorted(set(required_suvr + required_volume).difference(data.columns))
    if missing:
        raise ConfigurationError(f"Missing SUVR/VOLUME columns for feature aggregation: {missing}")

    values: dict[str, np.ndarray] = {}
    valid_counts: dict[str, np.ndarray] = {}
    total_volumes: dict[str, np.ndarray] = {}

    for feature, suvr_columns in feature_map.items():
        volume_columns = [suvr_to_volume_column(column) for column in suvr_columns]
        suvr = data[suvr_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        volume = data[volume_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(suvr) & np.isfinite(volume) & (volume > 0)
        weighted_sum = np.where(valid, suvr * volume, 0.0).sum(axis=1)
        volume_sum = np.where(valid, volume, 0.0).sum(axis=1)
        count = valid.sum(axis=1)
        output = np.full(len(data), np.nan, dtype=float)
        usable = (count >= minimum_valid_rois) & (volume_sum > 0)
        output[usable] = weighted_sum[usable] / volume_sum[usable]
        values[feature] = output
        valid_counts[f"{feature}_valid_roi_count"] = count
        total_volumes[f"{feature}_total_volume"] = volume_sum

    return FeatureAggregationResult(
        values=pd.DataFrame(values, index=data.index),
        valid_roi_counts=pd.DataFrame(valid_counts, index=data.index),
        total_volumes=pd.DataFrame(total_volumes, index=data.index),
    )

