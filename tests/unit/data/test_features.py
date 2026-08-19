import numpy as np
import pandas as pd
import pytest

from storm_pet.data.features import aggregate_volume_weighted_features
from storm_pet.exceptions import ConfigurationError


def test_volume_weighted_feature_matches_hand_calculation() -> None:
    data = pd.DataFrame(
        {
            "A_SUVR": [1.0, 5.0],
            "A_VOLUME": [1.0, 0.0],
            "B_SUVR": [3.0, 7.0],
            "B_VOLUME": [3.0, 2.0],
        }
    )
    result = aggregate_volume_weighted_features(data, {"feature": ["A_SUVR", "B_SUVR"]})
    assert result.values["feature"].tolist() == pytest.approx([2.5, 7.0])
    assert result.valid_roi_counts["feature_valid_roi_count"].tolist() == [2, 1]
    assert result.total_volumes["feature_total_volume"].tolist() == pytest.approx([4.0, 2.0])


def test_volume_weighted_feature_respects_minimum_valid_rois() -> None:
    data = pd.DataFrame(
        {
            "A_SUVR": [1.0],
            "A_VOLUME": [0.0],
            "B_SUVR": [3.0],
            "B_VOLUME": [2.0],
        }
    )
    result = aggregate_volume_weighted_features(
        data, {"feature": ["A_SUVR", "B_SUVR"]}, minimum_valid_rois=2
    )
    assert np.isnan(result.values.loc[0, "feature"])


def test_volume_weighted_feature_requires_matching_volume_columns() -> None:
    with pytest.raises(ConfigurationError, match="Missing SUVR/VOLUME"):
        aggregate_volume_weighted_features(
            pd.DataFrame({"A_SUVR": [1.0]}), {"feature": ["A_SUVR"]}
        )

