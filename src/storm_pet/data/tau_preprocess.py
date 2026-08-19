from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from storm_pet.exceptions import ConfigurationError

DEFAULT_KEYS = ["PTID", "VISCODE", "VISCODE2"]
DEFAULT_OPTIONAL_ABETA_COLUMNS = [
    "SUMMARY_SUVR",
    "CENTILOIDS",
    "ABETA_CL_THRESHOLD",
]
DEFAULT_TAU_PASSTHROUGH_COLUMNS = [
    "TRACER",
    "TRACER_SUVR_WARNING",
    "ABETA_CL_LABEL",
    "ABETA_CL_LABEL_monotonic",
    "ABETA_MATCHED_DATE_180D",
    "ABETA_DATE_DIFF_DAYS_180D",
    "CENTILOIDS_NEAREST_180D",
    "ABETA_CL_THRESHOLD_NEAREST_180D",
]


def normalize_keys(
    data: pd.DataFrame, keys: Iterable[str] = DEFAULT_KEYS
) -> pd.DataFrame:
    output = data.copy()
    for column in keys:
        if column not in output.columns:
            raise ConfigurationError(f"Missing key column: {column}")
        output[column] = output[column].astype("string").str.strip()
    return output


def get_suvr_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column.endswith("_SUVR")]


def get_volume_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column.endswith("_VOLUME")]


def require_unique_keys(data: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicate = data.duplicated(keys, keep=False)
    if duplicate.any():
        examples = data.loc[duplicate, keys].head(10).to_dict(orient="records")
        raise ConfigurationError(
            f"{name} contains duplicate rows on {keys}; examples={examples}"
        )


def interleave_suvr_volume_columns(data: pd.DataFrame) -> pd.DataFrame:
    volume_columns = get_volume_columns(data)
    volume_set = set(volume_columns)
    used: set[str] = set()
    ordered: list[str] = []
    for column in data.columns:
        if column in volume_set:
            continue
        ordered.append(column)
        if column.endswith("_SUVR"):
            volume_column = f"{column[:-5]}_VOLUME"
            if volume_column in volume_set:
                ordered.append(volume_column)
                used.add(volume_column)
    ordered.extend(column for column in volume_columns if column not in used)
    return data[ordered]


def merge_diagnosis(
    tau_data: pd.DataFrame,
    diagnosis_data: pd.DataFrame,
    keys: list[str] | None = None,
    diagnosis_column: str = "DIAGNOSIS",
) -> pd.DataFrame:
    keys = list(keys or DEFAULT_KEYS)
    tau = normalize_keys(tau_data, keys)
    diagnosis = normalize_keys(diagnosis_data, keys)
    if diagnosis_column not in diagnosis.columns:
        raise ConfigurationError(f"Diagnosis data is missing {diagnosis_column}")
    mapping = diagnosis[keys + [diagnosis_column]].copy()
    require_unique_keys(mapping, keys, "diagnosis_data")
    output = tau.merge(mapping, on=keys, how="left", validate="many_to_one")
    output = output.rename(columns={diagnosis_column: "diagnosis"})
    output["diagnosis"] = pd.to_numeric(output["diagnosis"], errors="coerce").astype(
        "Int64"
    )
    output["group"] = output["diagnosis"].map({1: "CN", 2: "MCI", 3: "AD"})
    return output


def merge_tau_with_nearest_abeta(
    tau_data: pd.DataFrame,
    abeta_data: pd.DataFrame,
    max_difference_days: int = 180,
    id_column: str = "PTID",
    tau_date_column: str = "SCANDATE",
    abeta_date_column: str = "SCANDATE",
    abeta_label_column: str = "ABETA_CL_LABEL",
    optional_abeta_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Match each Tau scan to the closest Aβ scan; an exact tie selects the earlier Aβ date."""
    if max_difference_days < 0:
        raise ConfigurationError("max_difference_days must be non-negative")
    optional_abeta_columns = list(
        DEFAULT_OPTIONAL_ABETA_COLUMNS
        if optional_abeta_columns is None
        else optional_abeta_columns
    )
    tau = tau_data.copy()
    abeta = abeta_data.copy()
    for data, column, name in [
        (tau, tau_date_column, "Tau"),
        (abeta, abeta_date_column, "Aβ"),
    ]:
        if id_column not in data.columns or column not in data.columns:
            raise ConfigurationError(f"{name} data is missing {id_column} or {column}")
    if abeta_label_column not in abeta.columns:
        raise ConfigurationError(f"Aβ data is missing {abeta_label_column}")

    tau[tau_date_column] = pd.to_datetime(tau[tau_date_column], errors="coerce")
    abeta[abeta_date_column] = pd.to_datetime(abeta[abeta_date_column], errors="coerce")
    renamed_date = "ABETA_SCANDATE"
    abeta = abeta.rename(columns={abeta_date_column: renamed_date})
    keep = [id_column, renamed_date, abeta_label_column]
    keep.extend(column for column in optional_abeta_columns if column in abeta.columns)
    abeta = abeta[list(dict.fromkeys(keep))]

    valid_tau = tau[tau[id_column].notna() & tau[tau_date_column].notna()].copy()
    valid_abeta = abeta[abeta[id_column].notna() & abeta[renamed_date].notna()].copy()
    valid_tau = valid_tau.reset_index(names="_tau_original_index")
    candidates = valid_tau.merge(valid_abeta, on=id_column, how="left")
    candidates["ABETA_DATE_DIFF_DAYS"] = (
        candidates[tau_date_column] - candidates[renamed_date]
    ).abs().dt.days
    nearest = (
        candidates.sort_values(
            ["_tau_original_index", "ABETA_DATE_DIFF_DAYS", renamed_date],
            na_position="last",
        )
        .groupby("_tau_original_index", as_index=False)
        .first()
    )

    over_limit = nearest["ABETA_DATE_DIFF_DAYS"].isna() | (
        nearest["ABETA_DATE_DIFF_DAYS"] > max_difference_days
    )
    suffix = f"NEAREST_{max_difference_days}D"
    label_output = f"{abeta_label_column}_{suffix}"
    nearest[label_output] = nearest[abeta_label_column]
    nearest.loc[over_limit, label_output] = np.nan
    date_output = f"ABETA_MATCHED_DATE_{max_difference_days}D"
    difference_output = f"ABETA_DATE_DIFF_DAYS_{max_difference_days}D"
    nearest[date_output] = nearest[renamed_date]
    nearest.loc[over_limit, date_output] = pd.NaT
    nearest[difference_output] = nearest["ABETA_DATE_DIFF_DAYS"]
    nearest.loc[over_limit, difference_output] = np.nan

    output_columns = ["_tau_original_index", label_output, date_output, difference_output]
    for column in optional_abeta_columns:
        if column in nearest.columns:
            output_column = f"{column}_{suffix}"
            nearest[output_column] = nearest[column]
            nearest.loc[over_limit, output_column] = np.nan
            output_columns.append(output_column)

    output = tau.reset_index(names="_tau_original_index")
    output = output.merge(nearest[output_columns], on="_tau_original_index", how="left")
    return output.drop(columns="_tau_original_index")


def standardize_abeta_label_column(
    data: pd.DataFrame,
    max_difference_days: int = 180,
    abeta_label_column: str = "ABETA_CL_LABEL",
) -> pd.DataFrame:
    output = data.copy()
    nearest_column = f"{abeta_label_column}_NEAREST_{max_difference_days}D"
    source = nearest_column if nearest_column in output.columns else abeta_label_column
    if source not in output.columns:
        raise ConfigurationError(
            f"Cannot find {abeta_label_column} or {nearest_column}"
        )
    output[abeta_label_column] = pd.to_numeric(output[source], errors="coerce")
    return output


def build_demographic_tau_table(
    tau_abeta_data: pd.DataFrame,
    demographic_data: pd.DataFrame | None = None,
    keys: list[str] | None = None,
    passthrough_columns: list[str] | None = None,
) -> pd.DataFrame:
    keys = list(keys or DEFAULT_KEYS)
    passthrough_columns = list(
        DEFAULT_TAU_PASSTHROUGH_COLUMNS
        if passthrough_columns is None
        else passthrough_columns
    )
    source = normalize_keys(tau_abeta_data, keys)
    if "Research Group" not in source.columns and "group" in source.columns:
        source = source.copy()
        source["Research Group"] = source["group"]
    if demographic_data is None:
        return source.copy()

    target = normalize_keys(demographic_data, keys)
    require_unique_keys(source, keys, "tau_abeta_data")
    require_unique_keys(target, keys, "demographic_data")
    suvr_columns = get_suvr_columns(source)
    volume_columns = get_volume_columns(source)
    old_region_prefixes = ("ctx-", "left-", "right-", "brain-")
    keep = [
        column
        for column in target.columns
        if not column.endswith("_SUVR")
        and not column.endswith("_VOLUME")
        and not column.startswith(old_region_prefixes)
    ]
    keep = list(dict.fromkeys(keep))
    passthrough = [
        column
        for column in passthrough_columns
        if column in source.columns and column not in keys and column not in keep
    ]
    source_columns = list(dict.fromkeys(keys + passthrough + suvr_columns + volume_columns))
    output = target[keep].merge(
        source[source_columns], on=keys, how="left", validate="one_to_one"
    )
    output_columns = keep + passthrough + [
        column for column in suvr_columns + volume_columns if column in output.columns
    ]
    return interleave_suvr_volume_columns(output[output_columns])


def find_abeta_time_violations(data: pd.DataFrame, label_column: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    ordered = data.sort_values(["PTID", "SCANDATE", "orig_index"])
    for ptid, group in ordered.groupby("PTID", sort=False):
        first_positive_date = None
        for _, row in group.iterrows():
            value = row[label_column]
            if pd.isna(value):
                continue
            if float(value) == 1.0 and first_positive_date is None:
                first_positive_date = row["SCANDATE"]
            elif float(value) == 0.0 and first_positive_date is not None:
                records.append(
                    {
                        "PTID": ptid,
                        "first_positive_date": first_positive_date,
                        "later_negative_date": row["SCANDATE"],
                    }
                )
                break
    return pd.DataFrame(
        records, columns=["PTID", "first_positive_date", "later_negative_date"]
    )


def apply_abeta_monotonic_correction(
    data: pd.DataFrame,
    label_column: str = "ABETA_CL_LABEL",
    date_column: str = "SCANDATE",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = data.copy().reset_index(names="orig_index")
    output[date_column] = pd.to_datetime(output[date_column], errors="coerce")
    output[label_column] = pd.to_numeric(output[label_column], errors="coerce")
    output = output.sort_values(["PTID", date_column, "orig_index"]).reset_index(drop=True)
    original_column = f"{label_column}_original"
    monotonic_column = f"{label_column}_monotonic"
    output[original_column] = output[label_column]
    output[monotonic_column] = output[label_column]
    output["modify_reason"] = "original"
    before = find_abeta_time_violations(output, original_column)
    changed_ptids: list[object] = []

    for ptid, indices in output.groupby("PTID", sort=False).groups.items():
        indices = list(indices)
        group = output.loc[indices].reset_index(names="global_index")
        labels = group[original_column].tolist()
        first_positive = next(
            (i for i, value in enumerate(labels) if pd.notna(value) and float(value) == 1.0),
            None,
        )
        changed = False
        if first_positive is not None:
            for i in range(first_positive, len(labels)):
                global_index = int(group.loc[i, "global_index"])
                value = labels[i]
                if pd.isna(value):
                    output.at[global_index, monotonic_column] = 1.0
                    output.at[global_index, "modify_reason"] = (
                        "filled_to_1_after_first_positive"
                    )
                    changed = True
                elif float(value) == 0.0:
                    output.at[global_index, monotonic_column] = 1.0
                    output.at[global_index, "modify_reason"] = (
                        "corrected_0_to_1_after_first_positive"
                    )
                    changed = True

        prefix_end = first_positive if first_positive is not None else len(labels)
        seen_negative = False
        negative_exists_later = [False] * prefix_end
        for i in range(prefix_end - 1, -1, -1):
            negative_exists_later[i] = seen_negative
            value = labels[i]
            if pd.notna(value) and float(value) == 0.0:
                seen_negative = True
        for i in range(prefix_end):
            if pd.isna(labels[i]) and negative_exists_later[i]:
                global_index = int(group.loc[i, "global_index"])
                output.at[global_index, monotonic_column] = 0.0
                output.at[global_index, "modify_reason"] = (
                    "filled_to_0_before_first_positive_from_later_negative"
                )
                changed = True
        if changed:
            changed_ptids.append(ptid)

    after = find_abeta_time_violations(output, monotonic_column)
    corrected = output[output["PTID"].isin(changed_ptids)].copy()
    corrected = corrected.sort_values(["PTID", date_column, "orig_index"])
    output = output.sort_values("orig_index").drop(columns="orig_index").reset_index(drop=True)
    corrected = corrected.drop(columns="orig_index").reset_index(drop=True)
    return output, before, after, corrected


def _set_group_value(
    corrected: list[object],
    reasons: list[str],
    position: int,
    new_value: str,
    missing_reason: str,
    correction_reason: str,
) -> None:
    old_value = corrected[position]
    if pd.isna(old_value):
        corrected[position] = new_value
        reasons[position] = missing_reason
    elif str(old_value) != new_value:
        corrected[position] = new_value
        reasons[position] = correction_reason


def apply_research_group_correction(
    data: pd.DataFrame,
    group_column: str = "Research Group",
    date_column: str = "SCANDATE",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = data.copy().reset_index(names="orig_index")
    output[date_column] = pd.to_datetime(output[date_column], errors="coerce")
    if group_column not in output.columns:
        if "group" not in output.columns:
            raise ConfigurationError("Cannot find Research Group or group column")
        group_column = "group"
    original_column = f"{group_column}_original"
    corrected_column = f"{group_column}_corrected"
    reason_column = f"{group_column}_modify_reason"
    output[original_column] = output[group_column].where(
        output[group_column].isin({"CN", "MCI", "AD"}), np.nan
    )
    output[corrected_column] = output[original_column]
    output[reason_column] = "original"
    reports: list[dict[str, object]] = []
    changed_ptids: list[object] = []

    ordered = output.sort_values(["PTID", date_column, "orig_index"])
    for ptid, indices in ordered.groupby("PTID", sort=False).groups.items():
        group = output.loc[list(indices)].sort_values([date_column, "orig_index"])
        original = group[original_column].tolist()
        corrected = original.copy()
        reasons = ["original"] * len(original)
        positions = {
            label: [i for i, value in enumerate(original) if value == label]
            for label in ("CN", "MCI", "AD")
        }
        if len(positions["MCI"]) >= 2:
            for i in range(min(positions["MCI"]), max(positions["MCI"]) + 1):
                _set_group_value(
                    corrected,
                    reasons,
                    i,
                    "MCI",
                    "filled_to_MCI_between_observed_MCI",
                    "corrected_to_MCI_between_observed_MCI",
                )
        if positions["CN"]:
            for i in range(max(positions["CN"]) + 1):
                _set_group_value(
                    corrected,
                    reasons,
                    i,
                    "CN",
                    "filled_to_CN_before_later_CN",
                    "corrected_to_CN_before_later_CN",
                )
        if positions["AD"]:
            for i in range(min(positions["AD"]), len(original)):
                _set_group_value(
                    corrected,
                    reasons,
                    i,
                    "AD",
                    "filled_to_AD_after_earlier_AD",
                    "corrected_to_AD_after_earlier_AD",
                )

        local_changed = False
        for local_index, global_index in enumerate(group.index):
            output.at[global_index, corrected_column] = corrected[local_index]
            output.at[global_index, reason_column] = reasons[local_index]
            if reasons[local_index] != "original":
                local_changed = True
                reports.append(
                    {
                        "PTID": ptid,
                        "SCANDATE": group.iloc[local_index][date_column],
                        "VISCODE": group.iloc[local_index].get("VISCODE", np.nan),
                        "VISCODE2": group.iloc[local_index].get("VISCODE2", np.nan),
                        "original_group": original[local_index],
                        "corrected_group": corrected[local_index],
                        "reason": reasons[local_index],
                    }
                )
        if local_changed:
            changed_ptids.append(ptid)

    output[group_column] = output[corrected_column]
    if group_column == "Research Group" and "group" in output.columns:
        output["group"] = output[corrected_column]
    if "diagnosis" in output.columns:
        output["diagnosis_corrected"] = output[corrected_column].map(
            {"CN": 1, "MCI": 2, "AD": 3}
        ).astype("Int64")
    report = pd.DataFrame(
        reports,
        columns=[
            "PTID",
            "SCANDATE",
            "VISCODE",
            "VISCODE2",
            "original_group",
            "corrected_group",
            "reason",
        ],
    )
    corrected_trajectories = output[output["PTID"].isin(changed_ptids)].copy()
    corrected_trajectories = corrected_trajectories.sort_values(
        ["PTID", date_column, "orig_index"]
    )
    output = output.sort_values("orig_index").drop(columns="orig_index").reset_index(drop=True)
    corrected_trajectories = corrected_trajectories.drop(columns="orig_index").reset_index(
        drop=True
    )
    return output, report, corrected_trajectories


def drop_missing_research_group(
    data: pd.DataFrame, group_column: str = "Research Group"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_column not in data.columns:
        if "group" not in data.columns:
            raise ConfigurationError("Cannot find Research Group or group column")
        group_column = "group"
    missing = data[group_column].isna()
    return (
        data.loc[~missing].copy().reset_index(drop=True),
        data.loc[missing].copy().reset_index(drop=True),
    )


def filter_tracer(
    data: pd.DataFrame, tracer: str, tracer_column: str = "TRACER"
) -> pd.DataFrame:
    if tracer_column not in data.columns:
        raise ConfigurationError(f"Missing tracer column: {tracer_column}")
    normalized = data[tracer_column].astype("string").str.strip().str.upper()
    return data.loc[normalized == tracer.strip().upper()].copy().reset_index(drop=True)


def clean_for_sustain(
    data: pd.DataFrame,
    monotonic_label_column: str = "ABETA_CL_LABEL_monotonic",
    max_row_missing_suvr: int | None = 5,
    max_column_missing_rate: float | None = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = data.copy()
    if monotonic_label_column not in output.columns:
        raise ConfigurationError(f"Missing monotonic label column: {monotonic_label_column}")
    suvr_columns = get_suvr_columns(output)
    report: list[dict[str, object]] = []
    original_count = len(output)
    output = output[output[monotonic_label_column].notna()].copy()
    report.append(
        {
            "step": "drop_missing_monotonic_abeta",
            "removed": original_count - len(output),
            "remaining": len(output),
        }
    )
    if max_column_missing_rate is not None:
        missing_rate = output[suvr_columns].isna().mean()
        drop_columns = missing_rate[missing_rate > max_column_missing_rate].index.tolist()
        paired_volumes = [f"{column[:-5]}_VOLUME" for column in drop_columns]
        paired_volumes = [column for column in paired_volumes if column in output.columns]
        output = output.drop(columns=drop_columns + paired_volumes)
        suvr_columns = [column for column in suvr_columns if column not in drop_columns]
        report.append(
            {
                "step": f"drop_suvr_columns_missing_rate_gt_{max_column_missing_rate}",
                "removed": len(drop_columns),
                "remaining": len(suvr_columns),
            }
        )
        if paired_volumes:
            report.append(
                {
                    "step": "drop_volume_columns_paired_with_removed_suvr",
                    "removed": len(paired_volumes),
                    "remaining": len(get_volume_columns(output)),
                }
            )
    if max_row_missing_suvr is not None:
        before = len(output)
        row_missing = output[suvr_columns].isna().sum(axis=1)
        output = output[row_missing <= max_row_missing_suvr].copy()
        report.append(
            {
                "step": f"drop_rows_suvr_missing_gt_{max_row_missing_suvr}",
                "removed": before - len(output),
                "remaining": len(output),
            }
        )
    return interleave_suvr_volume_columns(output).reset_index(drop=True), pd.DataFrame(report)
