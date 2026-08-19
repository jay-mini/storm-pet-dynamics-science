"""Canonical Centiloid-based Aβ status preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from storm_pet.exceptions import ConfigurationError


CENTILOID_COLUMN = "CENTILOIDS"
ABETA_LABEL_COLUMN = "ABETA_CL_LABEL"
ABETA_THRESHOLD_COLUMN = "ABETA_CL_THRESHOLD"
CENTILOID_THRESHOLD = 18.0


def extract_roi_suvr_columns(
    data: pd.DataFrame, expected_count: int = 163
) -> list[str]:
    """Return the expected regional Aβ SUVR columns, excluding SUMMARY_SUVR."""
    columns = [column for column in data.columns if column.endswith("_SUVR")]
    roi_columns = [column for column in columns if column != "SUMMARY_SUVR"]
    if len(roi_columns) != expected_count:
        raise ConfigurationError(
            f"Expected {expected_count} ROI SUVR columns after excluding SUMMARY_SUVR, "
            f"found {len(roi_columns)}"
        )
    return roi_columns


def assign_centiloid_status(
    data: pd.DataFrame,
    *,
    centiloid_column: str = CENTILOID_COLUMN,
    label_column: str = ABETA_LABEL_COLUMN,
    threshold_column: str = ABETA_THRESHOLD_COLUMN,
    threshold: float = CENTILOID_THRESHOLD,
) -> pd.DataFrame:
    """Assign Aβ positivity using the fixed strict rule ``CENTILOIDS > 18``."""
    if centiloid_column not in data.columns:
        raise ConfigurationError(f"Missing required Centiloid column: {centiloid_column}")

    output = data.copy()
    centiloids = pd.to_numeric(output[centiloid_column], errors="coerce")
    labels = pd.Series(pd.NA, index=output.index, dtype="Int64")
    valid = centiloids.notna()
    labels.loc[valid] = (centiloids.loc[valid] > threshold).astype(int)
    output[label_column] = labels
    output[threshold_column] = float(threshold)
    return output
