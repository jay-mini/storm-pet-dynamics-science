import numpy as np

from storm_pet.sustain.artifacts import SustainBundle
from storm_pet.sustain.inference import infer_new_data
from storm_pet.sustain.preprocessing import fit_sustain_preprocessor


class RecordingBackend:
    def __init__(self) -> None:
        self.data = None
        self.n_samples = None

    def subtype_and_stage_individuals_newData(self, data, sequence, fractions, n_samples):
        self.data = data
        self.n_samples = n_samples
        n = data.shape[0]
        return (
            np.zeros((n, 1)),
            np.ones((n, 1)),
            np.zeros((n, 1)),
            np.ones((n, 1)),
            np.ones((n, 1)),
            np.ones((n, 2)) / 2,
            np.ones((n, 2, 1)) / 2,
        )


def test_inference_applies_frozen_normalization_and_caps_sample_count(tmp_path) -> None:
    train = np.array([[0.0], [2.0], [4.0], [6.0]])
    preprocessor = fit_sustain_preprocessor(
        train,
        control_mask=np.array([True, True, False, False]),
        disease_mask=np.array([False, False, True, True]),
        feature_names=["roi"],
        event_zscores=[1.0],
        minimum_supporting_scans=2,
    )
    bundle = SustainBundle(
        root=tmp_path,
        model_id="test",
        modality="tau",
        selected_subtypes=1,
        preprocessor=preprocessor,
        samples_sequence=np.array([[[0, 0, 0]]]),
        samples_f=np.array([[1.0, 1.0, 1.0]]),
    )
    backend = RecordingBackend()
    result = infer_new_data(bundle, np.array([[3.0], [5.0]]), ["roi"], backend)

    np.testing.assert_allclose(backend.data, [[2.0], [4.0]])
    assert backend.n_samples == 3
    assert result.ml_subtype.shape == (2, 1)
