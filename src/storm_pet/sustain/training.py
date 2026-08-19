from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from storm_pet.config import config_sha256, load_yaml, require_keys
from storm_pet.data.features import aggregate_volume_weighted_features, load_feature_map
from storm_pet.exceptions import ConfigurationError
from storm_pet.provenance import environment_summary, sha256_file, write_json_atomic
from storm_pet.sustain.preprocessing import SustainPreprocessor, fit_sustain_preprocessor


@dataclass(frozen=True)
class PreparedSustainTraining:
    z_data: np.ndarray
    preprocessor: SustainPreprocessor
    input_rows: int
    retained_rows: int
    control_rows: int
    disease_rows: int
    aggregation_mode: str
    retained_row_numbers: tuple[int, ...]


def _normalized(values: pd.Series) -> pd.Series:
    def normalize(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value).strip().casefold()

    return values.map(normalize)


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _training_contract(config: dict[str, Any]) -> dict[str, Any]:
    require_keys(
        config,
        {
            "schema_version",
            "model_id",
            "feature_map",
            "aggregation",
            "input",
            "control",
            "standardization",
            "features",
            "events",
            "training",
        },
        "SuStaIn config",
    )
    contract = config["input"]
    require_keys(
        contract,
        {
            "research_group_column",
            "abeta_status_column",
            "allowed_research_groups",
            "disease_research_groups",
            "abeta_negative_values",
        },
        "SuStaIn input contract",
    )
    return contract


def prepare_sustain_training_data(
    input_csv: Path,
    config_path: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], PreparedSustainTraining]:
    """Prepare the paper's full-data z-score SuStaIn input without running MCMC."""

    config = load_yaml(config_path)
    contract = _training_contract(config)
    frame = pd.read_csv(input_csv)
    frame["__storm_source_row"] = np.arange(len(frame), dtype=int)
    input_rows = len(frame)
    group_column = str(contract["research_group_column"])
    status_column = str(contract["abeta_status_column"])
    missing_contract = {group_column, status_column}.difference(frame.columns)
    if missing_contract:
        raise ConfigurationError(
            f"SuStaIn input is missing contract columns: {sorted(missing_contract)}"
        )

    allowed_groups = {str(value).strip().casefold() for value in contract["allowed_research_groups"]}
    disease_groups = {
        str(value).strip().casefold() for value in contract["disease_research_groups"]
    }
    group_values = _normalized(frame[group_column])
    frame = frame.loc[group_values.isin(allowed_groups)].copy()
    group_values = _normalized(frame[group_column])

    feature_map_path = _resolve_path(config["feature_map"], repository_root)
    feature_map = load_feature_map(feature_map_path)
    feature_names = tuple(feature_map)
    if set(feature_names).issubset(frame.columns):
        features = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        aggregation_mode = "preaggregated"
    else:
        aggregation = config["aggregation"]
        features = aggregate_volume_weighted_features(
            frame,
            feature_map,
            minimum_valid_rois=int(aggregation["minimum_valid_rois"]),
        ).values
        aggregation_mode = "volume_weighted_mean"

    finite_rows = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    if bool(config["aggregation"].get("reject_scan_if_any_feature_missing", True)):
        frame = frame.loc[finite_rows].copy()
        features = features.loc[finite_rows].copy()
        group_values = _normalized(frame[group_column])
    elif not finite_rows.all():
        raise ConfigurationError(
            "SuStaIn training does not impute missing features; enable scan rejection"
        )

    negative_values = {
        str(value).strip().casefold() for value in contract["abeta_negative_values"]
    }
    status_values = _normalized(frame[status_column])
    control_group = str(config["control"]["research_group"]).strip().casefold()
    control_mask = (group_values == control_group) & status_values.isin(negative_values)
    disease_mask = group_values.isin(disease_groups)

    standardization = config["standardization"]
    events = config["events"]
    preprocessor = fit_sustain_preprocessor(
        features.to_numpy(dtype=float),
        control_mask.to_numpy(dtype=bool),
        disease_mask.to_numpy(dtype=bool),
        feature_names,
        events["z_scores"],
        minimum_supporting_scans=int(events["minimum_supporting_scans"]),
        drop_non_increasing_disease_mean=bool(
            config["features"]["drop_non_increasing_disease_mean"]
        ),
        ddof=int(standardization["ddof"]),
        clip_min=float(standardization["clip_min"]),
        clip_max=float(standardization["clip_max"]),
        terminal_zmax=float(events["terminal_zmax"]),
    )
    z_data = preprocessor.transform(features.to_numpy(dtype=float), feature_names)
    return config, PreparedSustainTraining(
        z_data=z_data,
        preprocessor=preprocessor,
        input_rows=input_rows,
        retained_rows=len(frame),
        control_rows=int(control_mask.sum()),
        disease_rows=int(disease_mask.sum()),
        aggregation_mode=aggregation_mode,
        retained_row_numbers=tuple(frame["__storm_source_row"].astype(int).tolist()),
    )


def _write_preprocessor(output_dir: Path, preprocessor: SustainPreprocessor) -> None:
    np.savez_compressed(
        output_dir / "preprocessor.npz",
        feature_mask=preprocessor.feature_mask,
        control_mean=preprocessor.control_mean,
        control_std=preprocessor.control_std,
        z_vals=preprocessor.z_vals,
        z_max=preprocessor.z_max,
    )
    metadata = preprocessor.metadata()
    metadata["input_features"] = list(preprocessor.input_features)
    metadata["final_features"] = list(preprocessor.final_features)
    write_json_atomic(output_dir / "preprocessor.json", metadata)


def run_sustain_training(
    *,
    input_csv: Path,
    config_path: Path,
    output_dir: Path,
    repository_root: Path,
    dataset_name: str | None = None,
    seed: int | None = None,
    use_parallel_startpoints: bool = True,
) -> Path:
    """Train the main-text full-data model; cross-validation is intentionally separate."""

    config, prepared = prepare_sustain_training_data(
        input_csv.resolve(), config_path.resolve(), repository_root.resolve()
    )
    try:
        import pySuStaIn
    except ImportError as error:
        raise RuntimeError(
            "pySuStaIn is required for training; install the project's 'sustain' extra"
        ) from error

    output_dir = output_dir.resolve()
    native_dir = output_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    _write_preprocessor(output_dir, prepared.preprocessor)

    training = config["training"]
    resolved_seed = int(
        seed if seed is not None else config.get("cross_validation", {}).get("random_seed", 42)
    )
    resolved_dataset_name = dataset_name or str(config["model_id"])
    model = pySuStaIn.ZscoreSustain(
        prepared.z_data,
        prepared.preprocessor.z_vals,
        prepared.preprocessor.z_max,
        list(prepared.preprocessor.final_features),
        int(training["startpoints"]),
        int(training["maximum_subtypes"]),
        int(training["mcmc_iterations"]),
        str(native_dir),
        resolved_dataset_name,
        bool(use_parallel_startpoints),
        seed=resolved_seed,
    )
    model.run_sustain_algorithm()

    pickle_files = sorted(
        str(path.relative_to(output_dir)).replace("\\", "/")
        for path in native_dir.rglob("*.pickle")
    )
    manifest = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "dataset_name": resolved_dataset_name,
        "config_sha256": config_sha256(config),
        "input_sha256": sha256_file(input_csv),
        "input_rows": prepared.input_rows,
        "retained_rows": prepared.retained_rows,
        "control_rows": prepared.control_rows,
        "disease_rows": prepared.disease_rows,
        "aggregation_mode": prepared.aggregation_mode,
        "final_features": list(prepared.preprocessor.final_features),
        "stage_max": prepared.preprocessor.stage_max,
        "seed": resolved_seed,
        "use_parallel_startpoints": bool(use_parallel_startpoints),
        "native_pickle_files": pickle_files,
        "cross_validation_run": False,
        "environment": environment_summary(),
    }
    manifest_path = output_dir / "training_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path
