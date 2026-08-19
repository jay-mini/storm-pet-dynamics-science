from __future__ import annotations

import numpy as np
import pytest

from storm_pet.exceptions import ArtifactValidationError
from storm_pet.ot_cfm.artifacts import export_ot_cfm_bundle, load_ot_cfm_bundle


def _export(root):
    return export_ot_cfm_bundle(
        root,
        model_id="test-model",
        modality="abeta",
        roi_columns=[f"ROI_{index}_SUVR" for index in range(68)],
        roi_mean=np.ones(68),
        roi_std=np.ones(68),
        time_grid=np.linspace(0, 2, 9),
        stage_to_bin={0: 0, 1: 1, 2: 2},
        autoencoder_architecture={"input_dim": 68, "latent_dim": 2, "hidden_width": 4, "n_subtypes": 2, "n_bins": 3},
        cfm_architecture={"latent_dim": 2, "hidden_width": 4, "n_subtypes": 2},
        autoencoder_state={"weight": np.ones((2, 2), dtype=np.float32)},
        cfm_state={"weight": np.zeros((2, 2), dtype=np.float32)},
        source={"purpose": "unit test"},
    )


def test_numeric_bundle_round_trip(tmp_path) -> None:
    exported = _export(tmp_path / "deploy")
    loaded = load_ot_cfm_bundle(exported.root)

    assert loaded.model_id == "test-model"
    assert loaded.final_bin == 2
    assert len(loaded.roi_columns) == 68
    assert loaded.stage_to_bin == {0: 0, 1: 1, 2: 2}


def test_bundle_rejects_tampered_numeric_artifact(tmp_path) -> None:
    exported = _export(tmp_path / "deploy")
    with (exported.root / "time_grid.npy").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        load_ot_cfm_bundle(exported.root)
