from pathlib import Path

import numpy as np

from storm_pet.sustain.artifacts import SustainBundle
from storm_pet.sustain.preprocessing import fit_sustain_preprocessor
from storm_pet.sustain.zscore_backend import (
    PreparedZScoreSustainBackend,
    ZScoreSustainInferenceBackend,
)


def _bundle(tmp_path: Path) -> SustainBundle:
    preprocessor = fit_sustain_preprocessor(
        np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 4.0], [5.0, 6.0]]),
        control_mask=np.array([True, True, False, False]),
        disease_mask=np.array([False, False, True, True]),
        feature_names=["a", "b"],
        event_zscores=[1.0],
        minimum_supporting_scans=2,
    )
    return SustainBundle(
        root=tmp_path,
        model_id="synthetic-v1",
        modality="tau",
        selected_subtypes=2,
        preprocessor=preprocessor,
        samples_sequence=np.array([[[0, 1, 0], [1, 0, 1]], [[1, 0, 1], [0, 1, 0]]]),
        samples_f=np.array([[0.65, 0.55, 0.6], [0.35, 0.45, 0.4]]),
    )


def test_prepared_backend_matches_reference_backend(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    data = np.array([[0.1, 0.2], [2.0, 3.0]])
    reference = ZScoreSustainInferenceBackend.from_bundle(
        bundle
    ).subtype_and_stage_individuals_newData(
        data, bundle.samples_sequence, bundle.samples_f, 3
    )
    prepared = PreparedZScoreSustainBackend.from_bundle(
        bundle, n_samples=3, posterior_chunk_size=2
    ).subtype_and_stage_individuals_newData(data)

    for expected, actual in zip(reference, prepared, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
