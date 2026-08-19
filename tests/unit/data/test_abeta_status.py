import numpy as np
import pandas as pd
import pytest

from storm_pet.data.abeta_status import assign_centiloid_status, extract_roi_suvr_columns
from storm_pet.exceptions import ConfigurationError


def test_centiloid_status_uses_strict_gt_18_and_preserves_missing() -> None:
    source = pd.DataFrame({"CENTILOIDS": [17, 18, 19, np.nan, "bad"]})
    result = assign_centiloid_status(source)

    assert result["ABETA_CL_LABEL"].tolist() == [0, 0, 1, pd.NA, pd.NA]
    assert result["ABETA_CL_THRESHOLD"].tolist() == [18.0] * len(source)


def test_centiloid_status_requires_centiloid_column() -> None:
    with pytest.raises(ConfigurationError, match="Centiloid"):
        assign_centiloid_status(pd.DataFrame({"SUMMARY_SUVR": [1.0]}))


def test_roi_schema_excludes_summary_suvr_and_requires_exact_count() -> None:
    columns = {"SUMMARY_SUVR": [1.0]}
    columns.update({f"ROI_{index}_SUVR": [1.0] for index in range(163)})
    assert len(extract_roi_suvr_columns(pd.DataFrame(columns))) == 163
    with pytest.raises(ConfigurationError, match="Expected 163"):
        extract_roi_suvr_columns(pd.DataFrame({"SUMMARY_SUVR": [1.0]}))
