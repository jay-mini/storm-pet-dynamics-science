from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from storm_pet.data.abeta_gmm import assign_abeta_status, extract_roi_suvr_columns, fit_abeta_gmm
from storm_pet.data.tau_preprocess import (
    apply_abeta_monotonic_correction,
    apply_research_group_correction,
    build_demographic_tau_table,
    clean_for_sustain,
    drop_missing_research_group,
    filter_tracer,
    merge_diagnosis,
    merge_tau_with_nearest_abeta,
    standardize_abeta_label_column,
)
from storm_pet.provenance import sha256_file, write_csv_atomic, write_json_atomic


@dataclass(frozen=True)
class DataStageResult:
    final_table: Path
    manifest: Path
    metrics: dict[str, object]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def prepare_abeta_gmm_stage(input_csv: Path, output_directory: Path) -> DataStageResult:
    output_directory.mkdir(parents=True, exist_ok=True)
    data = _read_csv(input_csv)
    roi_columns = extract_roi_suvr_columns(data)
    fit = fit_abeta_gmm(data)
    result = assign_abeta_status(data, fit)
    status_path = output_directory / "abeta_scan_table.csv"
    roi_path = output_directory / "abeta_roi163_table.csv"
    write_csv_atomic(status_path, result)
    metadata = [
        column
        for column in ["PTID", "VISCODE", "VISCODE2", "SCANDATE", "diagnosis", "group"]
        if column in result.columns
    ]
    write_csv_atomic(roi_path, result[metadata + roi_columns])
    metrics = {
        "n_rows": len(result),
        "n_valid_summary_suvr": int(fit.labels.notna().sum()),
        "n_positive": int((fit.labels == 1).sum()),
        "n_negative": int((fit.labels == 0).sum()),
        "gmm_means": list(fit.means),
        "gmm_standard_deviations": list(fit.standard_deviations),
        "gmm_weights": list(fit.weights),
        "posterior_cutoff": fit.posterior_cutoff,
        "legacy_grid_cutoff": fit.legacy_grid_cutoff,
        "n_roi_suvr": len(roi_columns),
    }
    metrics_path = output_directory / "data_qc.json"
    write_json_atomic(metrics_path, metrics)
    manifest_path = output_directory / "stage_manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "stage": "data",
            "modality": "abeta",
            "completed": True,
            "generated_utc": datetime.now(UTC).isoformat(),
            "inputs": {"scan_table_sha256": sha256_file(input_csv)},
            "outputs": {
                status_path.name: sha256_file(status_path),
                roi_path.name: sha256_file(roi_path),
                metrics_path.name: sha256_file(metrics_path),
            },
        },
    )
    return DataStageResult(status_path, manifest_path, metrics)


def prepare_tau_stage(
    tau_csv: Path,
    diagnosis_csv: Path,
    abeta_status_csv: Path,
    demographic_csv: Path,
    output_directory: Path,
    tracer: str = "FTP",
    max_difference_days: int = 180,
) -> DataStageResult:
    output_directory.mkdir(parents=True, exist_ok=True)
    tau = _read_csv(tau_csv)
    diagnosis = _read_csv(diagnosis_csv)
    abeta = _read_csv(abeta_status_csv)
    demographic = _read_csv(demographic_csv)
    tau_diagnosis = merge_diagnosis(tau, diagnosis)
    tau_abeta = merge_tau_with_nearest_abeta(
        tau_diagnosis, abeta, max_difference_days=max_difference_days
    )
    tau_abeta = standardize_abeta_label_column(
        tau_abeta, max_difference_days=max_difference_days
    )
    tau_demographic = build_demographic_tau_table(tau_abeta, demographic)
    tau_demographic, group_report, _ = apply_research_group_correction(tau_demographic)
    tau_demographic, missing_group = drop_missing_research_group(tau_demographic)
    tau_monotonic, violations_before, violations_after, _ = apply_abeta_monotonic_correction(
        tau_demographic
    )
    tau_clean, cleaning_report = clean_for_sustain(tau_monotonic)
    tau_tracer = filter_tracer(tau_clean, tracer)

    final_path = output_directory / "tau_scan_table.csv"
    write_csv_atomic(final_path, tau_tracer)
    write_csv_atomic(output_directory / "cleaning_report.csv", cleaning_report)
    metrics = {
        "raw_scans": len(tau),
        "after_diagnosis_merge": len(tau_diagnosis),
        "after_demographic_and_group_cleanup": len(tau_demographic),
        "after_sustain_cleaning": len(tau_clean),
        "selected_tracer": tracer,
        "selected_tracer_scans": len(tau_tracer),
        "selected_tracer_subjects": int(tau_tracer["PTID"].nunique()),
        "research_group_corrections": len(group_report),
        "missing_research_group_removed": len(missing_group),
        "abeta_time_violations_before": len(violations_before),
        "abeta_time_violations_after": len(violations_after),
        "n_suvr_columns": sum(column.endswith("_SUVR") for column in tau_tracer.columns),
        "n_volume_columns": sum(column.endswith("_VOLUME") for column in tau_tracer.columns),
    }
    metrics_path = output_directory / "data_qc.json"
    write_json_atomic(metrics_path, metrics)
    manifest_path = output_directory / "stage_manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "stage": "data",
            "modality": "tau",
            "completed": True,
            "generated_utc": datetime.now(UTC).isoformat(),
            "inputs": {
                "tau_sha256": sha256_file(tau_csv),
                "diagnosis_sha256": sha256_file(diagnosis_csv),
                "abeta_status_sha256": sha256_file(abeta_status_csv),
                "demographic_sha256": sha256_file(demographic_csv),
            },
            "outputs": {
                final_path.name: sha256_file(final_path),
                metrics_path.name: sha256_file(metrics_path),
            },
        },
    )
    return DataStageResult(final_path, manifest_path, metrics)

