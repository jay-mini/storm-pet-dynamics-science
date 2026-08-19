import json

import numpy as np
import pytest

from storm_pet.exceptions import ArtifactValidationError
from storm_pet.sustain.artifacts import (
    export_sustain_bundle,
    inventory_incoming_pickles,
    load_sustain_bundle,
)
from storm_pet.sustain.preprocessing import fit_sustain_preprocessor


def _preprocessor():
    values = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    return fit_sustain_preprocessor(
        values,
        control_mask=np.array([True, True, False, False]),
        disease_mask=np.array([False, False, True, True]),
        feature_names=["a", "b"],
        event_zscores=[1.0],
        minimum_supporting_scans=2,
    )


def test_safe_bundle_roundtrip_and_checksums(tmp_path) -> None:
    sequence = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    fractions = np.array([[0.6, 0.7], [0.4, 0.3]])
    export_sustain_bundle(
        tmp_path,
        model_id="synthetic-v1",
        modality="tau",
        selected_subtypes=2,
        preprocessor=_preprocessor(),
        samples_sequence=sequence,
        samples_f=fractions,
        source_pickle_sha256="a" * 64,
    )

    loaded = load_sustain_bundle(tmp_path)
    assert loaded.model_id == "synthetic-v1"
    assert loaded.preprocessor.final_features == ("a", "b")
    np.testing.assert_array_equal(loaded.samples_sequence, sequence)
    np.testing.assert_allclose(loaded.samples_f, fractions)
    assert set(json.loads((tmp_path / "manifest.json").read_text())["files"]) == {
        "metadata.json",
        "preprocessing.npz",
        "event_model.npz",
        "posterior_samples.npz",
    }

    with (tmp_path / "metadata.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        load_sustain_bundle(tmp_path)


def test_bundle_rejects_invalid_event_permutation(tmp_path) -> None:
    with pytest.raises(ArtifactValidationError, match="event permutation"):
        export_sustain_bundle(
            tmp_path,
            model_id="bad",
            modality="abeta",
            selected_subtypes=1,
            preprocessor=_preprocessor(),
            samples_sequence=np.array([[[0], [0]]]),
            samples_f=np.array([[1.0]]),
        )


def test_bundle_rejects_manifest_with_missing_checksum_entry(tmp_path) -> None:
    sequence = np.array([[[0], [1]], [[1], [0]]])
    export_sustain_bundle(
        tmp_path,
        model_id="synthetic-v1",
        modality="tau",
        selected_subtypes=2,
        preprocessor=_preprocessor(),
        samples_sequence=sequence,
        samples_f=np.array([[0.6], [0.4]]),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop("posterior_samples.npz")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="exact deploy artifacts"):
        load_sustain_bundle(tmp_path)


def test_pickle_inventory_never_deserializes(tmp_path) -> None:
    candidate = tmp_path / "model.pickle"
    candidate.write_bytes(b"not actually a pickle")
    records = inventory_incoming_pickles(tmp_path)
    assert records[0]["path"] == "model.pickle"
    assert records[0]["deserialized"] is False
    assert len(records[0]["sha256"]) == 64
