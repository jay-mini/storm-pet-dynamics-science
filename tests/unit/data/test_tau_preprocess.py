import numpy as np
import pandas as pd
import pytest

from storm_pet.data.tau_preprocess import (
    apply_abeta_monotonic_correction,
    apply_research_group_correction,
    merge_diagnosis,
    merge_tau_with_nearest_abeta,
)
from storm_pet.exceptions import ConfigurationError


def test_nearest_abeta_respects_window_and_tie_selects_earlier_date() -> None:
    tau = pd.DataFrame(
        {"PTID": ["A", "B"], "SCANDATE": ["2020-01-10", "2020-01-01"]}
    )
    abeta = pd.DataFrame(
        {
            "PTID": ["A", "A", "B"],
            "SCANDATE": ["2020-01-08", "2020-01-12", "2021-01-01"],
            "ABETA_CL_LABEL": [0, 1, 1],
        }
    )
    result = merge_tau_with_nearest_abeta(tau, abeta, max_difference_days=180)
    assert result.loc[0, "ABETA_CL_LABEL_NEAREST_180D"] == 0
    assert result.loc[0, "ABETA_MATCHED_DATE_180D"] == pd.Timestamp("2020-01-08")
    assert np.isnan(result.loc[1, "ABETA_CL_LABEL_NEAREST_180D"])


def test_abeta_monotonic_correction_matches_legacy_rules() -> None:
    data = pd.DataFrame(
        {
            "PTID": ["A"] * 5,
            "SCANDATE": pd.date_range("2020-01-01", periods=5),
            "ABETA_CL_LABEL": [np.nan, 0, np.nan, 1, 0],
        }
    )
    corrected, before, after, _ = apply_abeta_monotonic_correction(data)
    np.testing.assert_allclose(
        corrected["ABETA_CL_LABEL_monotonic"].to_numpy(float),
        [0, 0, np.nan, 1, 1],
        equal_nan=True,
    )
    assert len(before) == 1
    assert len(after) == 0


def test_research_group_correction_priority_is_monotonic() -> None:
    data = pd.DataFrame(
        {
            "PTID": ["A"] * 5,
            "SCANDATE": pd.date_range("2020-01-01", periods=5),
            "Research Group": ["MCI", np.nan, "MCI", "AD", "CN"],
        }
    )
    corrected, report, _ = apply_research_group_correction(data)
    assert corrected["Research Group"].tolist() == ["CN", "CN", "CN", "AD", "AD"]
    assert not report.empty


def test_duplicate_diagnosis_keys_fail_loudly() -> None:
    tau = pd.DataFrame({"PTID": ["A"], "VISCODE": ["v1"], "VISCODE2": ["v1"]})
    diagnosis = pd.DataFrame(
        {
            "PTID": ["A", "A"],
            "VISCODE": ["v1", "v1"],
            "VISCODE2": ["v1", "v1"],
            "DIAGNOSIS": [1, 2],
        }
    )
    with pytest.raises(ConfigurationError, match="duplicate"):
        merge_diagnosis(tau, diagnosis)
