from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from storm_pet.sustain.training import prepare_sustain_training_data
from storm_pet.sustain.training import run_sustain_training


def test_prepare_training_uses_configured_control_and_disease_groups(tmp_path: Path) -> None:
    feature_map = tmp_path / "features.yaml"
    feature_map.write_text(
        "schema_version: 1\nfeatures:\n  left: [LEFT_SUVR]\n  right: [RIGHT_SUVR]\n",
        encoding="utf-8",
    )
    config = tmp_path / "sustain.yaml"
    config.write_text(
        """schema_version: 1
model_id: test-model
feature_map: features.yaml
aggregation:
  minimum_valid_rois: 1
  reject_scan_if_any_feature_missing: true
input:
  research_group_column: Research Group
  abeta_status_column: amyloid
  allowed_research_groups: [CN, MCI, AD]
  disease_research_groups: [MCI, AD]
  abeta_negative_values: [0, negative]
control:
  research_group: CN
standardization: {ddof: 0, clip_min: -3, clip_max: 10}
features: {drop_non_increasing_disease_mean: true}
events: {z_scores: [1.0], terminal_zmax: 2.0, minimum_supporting_scans: 2}
training: {startpoints: 2, maximum_subtypes: 1, mcmc_iterations: 10}
""",
        encoding="utf-8",
    )
    source = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "Research Group": ["CN", "CN", "MCI", "AD"],
            "amyloid": [0, "negative", 1, 1],
            "left": [0.0, 2.0, 5.0, 6.0],
            "right": [1.0, 3.0, 6.0, 7.0],
        }
    ).to_csv(source, index=False)

    _, prepared = prepare_sustain_training_data(source, config, tmp_path)

    assert prepared.control_rows == 2
    assert prepared.disease_rows == 2
    assert prepared.retained_rows == 4
    assert prepared.aggregation_mode == "preaggregated"
    assert prepared.z_data.shape == (4, 2)
    np.testing.assert_allclose(prepared.preprocessor.z_max, [2.0, 2.0])


def test_training_writes_manifest_and_invokes_native_backend(
    tmp_path: Path, monkeypatch
) -> None:
    feature_map = tmp_path / "features.yaml"
    feature_map.write_text("features:\n  roi: [ROI_SUVR]\n", encoding="utf-8")
    config = tmp_path / "sustain.yaml"
    config.write_text(
        """schema_version: 1
model_id: synthetic-model
feature_map: features.yaml
aggregation: {minimum_valid_rois: 1, reject_scan_if_any_feature_missing: true}
input:
  research_group_column: group
  abeta_status_column: amyloid
  allowed_research_groups: [CN, MCI, AD]
  disease_research_groups: [MCI, AD]
  abeta_negative_values: [0]
control: {research_group: CN}
standardization: {ddof: 0, clip_min: -3, clip_max: 10}
features: {drop_non_increasing_disease_mean: true}
events: {z_scores: [1.0], terminal_zmax: 2.0, minimum_supporting_scans: 2}
training: {startpoints: 2, maximum_subtypes: 1, mcmc_iterations: 10}
cross_validation: {random_seed: 7}
""",
        encoding="utf-8",
    )
    source = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "group": ["CN", "CN", "MCI", "AD"],
            "amyloid": [0, 0, 1, 1],
            "roi": [0.0, 2.0, 5.0, 6.0],
        }
    ).to_csv(source, index=False)

    calls = {}

    class FakeZscoreSustain:
        def __init__(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs

        def run_sustain_algorithm(self):
            calls["ran"] = True

    monkeypatch.setitem(
        sys.modules,
        "pySuStaIn",
        SimpleNamespace(ZscoreSustain=FakeZscoreSustain),
    )
    manifest = run_sustain_training(
        input_csv=source,
        config_path=config,
        output_dir=tmp_path / "output",
        repository_root=tmp_path,
        use_parallel_startpoints=False,
    )

    assert manifest.is_file()
    assert (manifest.parent / "preprocessor.npz").is_file()
    assert calls["ran"] is True
    assert calls["kwargs"]["seed"] == 7
    assert calls["args"][5] == 1
