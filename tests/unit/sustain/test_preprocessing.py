import numpy as np
import pytest

from storm_pet.exceptions import ArtifactValidationError
from storm_pet.sustain.preprocessing import fit_sustain_preprocessor


def test_fit_uses_controls_and_freezes_supported_events() -> None:
    values = np.array(
        [
            [0.0, 4.0, 0.0],
            [2.0, 6.0, 2.0],
            [4.0, 3.0, 0.0],
            [6.0, 2.0, 0.0],
            [8.0, 1.0, 0.0],
        ]
    )
    fitted = fit_sustain_preprocessor(
        values,
        control_mask=np.array([True, True, False, False, False]),
        disease_mask=np.array([False, False, True, True, True]),
        feature_names=["increasing", "decreasing", "flat"],
        event_zscores=[1.0, 3.5],
        minimum_supporting_scans=2,
    )

    assert fitted.input_features == ("increasing", "decreasing", "flat")
    assert fitted.final_features == ("increasing",)
    np.testing.assert_array_equal(fitted.feature_mask, [True, False, False])
    np.testing.assert_allclose(fitted.control_mean, [1.0, 5.0, 1.0])
    np.testing.assert_allclose(fitted.control_std, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(fitted.z_vals, [[1.0, 3.5]])
    np.testing.assert_allclose(fitted.z_max, [5.0])
    assert fitted.stage_max == 2


def test_transform_rejects_feature_reordering_and_nonfinite_values() -> None:
    values = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    fitted = fit_sustain_preprocessor(
        values,
        control_mask=np.array([True, True, False, False]),
        disease_mask=np.array([False, False, True, True]),
        feature_names=["left", "right"],
        event_zscores=[1.0],
        minimum_supporting_scans=2,
    )

    with pytest.raises(ArtifactValidationError, match="names/order"):
        fitted.transform(values, ["right", "left"])
    bad = values.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ArtifactValidationError, match="NaN"):
        fitted.transform(bad, ["left", "right"])


def test_support_count_matches_legacy_strict_greater_than_rule() -> None:
    values = np.array([[0.0], [2.0], [3.0], [3.0]])
    fitted = fit_sustain_preprocessor(
        values,
        control_mask=np.array([True, True, False, False]),
        disease_mask=np.array([False, False, True, True]),
        feature_names=["roi"],
        event_zscores=[1.0, 2.0],
        minimum_supporting_scans=2,
    )
    # Control mean/std are 1/1, so standardized values are -1, 1, 2, 2.
    # The legacy code uses `> threshold`, not `>= threshold`; z=2 has no support.
    np.testing.assert_allclose(fitted.z_vals, [[1.0, 0.0]])
    np.testing.assert_allclose(fitted.z_max, [2.0])


def test_explicit_terminal_zmax_preserves_paper_contract() -> None:
    values = np.array([[0.0], [2.0], [5.0], [6.0]])
    fitted = fit_sustain_preprocessor(
        values,
        control_mask=np.array([True, True, False, False]),
        disease_mask=np.array([False, False, True, True]),
        feature_names=["roi"],
        event_zscores=[1.0, 3.5],
        minimum_supporting_scans=2,
        terminal_zmax=4.0,
    )
    np.testing.assert_allclose(fitted.z_max, [4.0])
