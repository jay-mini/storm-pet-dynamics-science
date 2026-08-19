from __future__ import annotations

import csv
from pathlib import Path

from storm_pet.exceptions import ConfigurationError


def load_roi_order(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["feature_index", "feature_name"]:
            raise ConfigurationError(
                f"ROI order must have feature_index,feature_name columns: {path}"
            )
        rows = list(reader)
    indices = [int(row["feature_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ConfigurationError(f"ROI indices must be contiguous and zero-based: {path}")
    names = [row["feature_name"].strip() for row in rows]
    if len(names) != len(set(names)):
        raise ConfigurationError(f"ROI order contains duplicate names: {path}")
    return names


def validate_ordered_columns(actual: list[str], expected: list[str]) -> None:
    if actual == expected:
        return
    missing = [name for name in expected if name not in actual]
    extra = [name for name in actual if name not in expected]
    first_mismatch = next(
        (
            index
            for index, (left, right) in enumerate(zip(actual, expected, strict=False))
            if left != right
        ),
        None,
    )
    raise ConfigurationError(
        "ROI columns do not match the registered order; "
        f"missing={missing}, extra={extra}, first_mismatch={first_mismatch}"
    )


def suvr_to_volume_column(suvr_column: str) -> str:
    if not suvr_column.endswith("_SUVR"):
        raise ConfigurationError(f"ROI column does not end with _SUVR: {suvr_column}")
    return f"{suvr_column[:-5]}_VOLUME"

