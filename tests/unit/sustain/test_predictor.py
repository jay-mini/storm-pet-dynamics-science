from pathlib import Path

import numpy as np
import pytest

from storm_pet.exceptions import ConfigurationError
from storm_pet.sustain.artifacts import SustainBundle
from storm_pet.sustain.predictor import SustainPredictor
from storm_pet.sustain.preprocessing import fit_sustain_preprocessor


def _predictor(tmp_path: Path) -> SustainPredictor:
    preprocessor = fit_sustain_preprocessor(
        np.array([[1.0], [1.2], [2.0], [2.4]]),
        control_mask=np.array([True, True, False, False]),
        disease_mask=np.array([False, False, True, True]),
        feature_names=["region"],
        event_zscores=[1.0],
        minimum_supporting_scans=2,
    )
    bundle = SustainBundle(
        root=tmp_path,
        model_id="synthetic-v1",
        modality="tau",
        selected_subtypes=1,
        preprocessor=preprocessor,
        samples_sequence=np.array([[[0, 0, 0]]]),
        samples_f=np.array([[1.0, 1.0, 1.0]]),
    )
    return SustainPredictor(
        bundle=bundle,
        feature_map={"region": ["ROI_SUVR"]},
        tracer="FTP",
        reference_region="inferior_cerebellar_gray",
        posterior_samples=3,
        max_batch_size=2,
    )


def test_predictor_aggregates_roi_values_and_returns_normalized_result(tmp_path) -> None:
    predictor = _predictor(tmp_path)
    prediction = predictor.predict_records(
        [("scan-1", {"ROI_SUVR": 1.8, "ROI_VOLUME": 1000.0})]
    )[0]

    assert prediction.scan_id == "scan-1"
    assert prediction.subtype == 0
    assert 0 <= prediction.stage <= 1
    assert sum(prediction.subtype_probabilities) == pytest.approx(1.0)
    assert sum(prediction.stage_probabilities) == pytest.approx(1.0)


def test_predictor_rejects_missing_columns_and_duplicate_ids(tmp_path) -> None:
    predictor = _predictor(tmp_path)
    with pytest.raises(ConfigurationError, match="missing required ROI columns"):
        predictor.predict_records([("scan-1", {"ROI_SUVR": 1.8})])
    with pytest.raises(ConfigurationError, match="unique"):
        predictor.predict_records(
            [
                ("scan-1", {"ROI_SUVR": 1.8, "ROI_VOLUME": 1000.0}),
                ("scan-1", {"ROI_SUVR": 1.9, "ROI_VOLUME": 1000.0}),
            ]
        )
