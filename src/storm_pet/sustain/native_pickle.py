from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from storm_pet.provenance import sha256_file
from storm_pet.sustain.artifacts import SustainBundle, export_sustain_bundle
from storm_pet.sustain.preprocessing import SustainPreprocessor


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Read a trusted pySuStaIn result dictionary without permitting arbitrary globals."""

    ALLOWED_GLOBALS = {
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
        ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
    }

    def find_class(self, module: str, name: str):
        try:
            return self.ALLOWED_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}") from error


def load_numpy_result(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        value = NumpyOnlyUnpickler(stream).load()
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise TypeError("SuStaIn pickle must contain a plain string-keyed dictionary")
    allowed = (np.ndarray, np.generic, int, float, bool, type(None))
    if any(not isinstance(item, allowed) for item in value.values()):
        raise TypeError("SuStaIn pickle contains a non-numeric value")
    return value


def load_training_preprocessor(root: Path) -> SustainPreprocessor:
    metadata = json.loads((root / "preprocessor.json").read_text(encoding="utf-8"))
    with np.load(root / "preprocessor.npz", allow_pickle=False) as arrays:
        return SustainPreprocessor(
            input_features=tuple(metadata["input_features"]),
            feature_mask=arrays["feature_mask"].astype(bool),
            final_features=tuple(metadata["final_features"]),
            control_mean=arrays["control_mean"],
            control_std=arrays["control_std"],
            z_vals=arrays["z_vals"],
            z_max=arrays["z_max"],
            clip_min=float(metadata["clip_min"]),
            clip_max=float(metadata["clip_max"]),
            ddof=int(metadata["ddof"]),
        )


def export_native_result(
    *,
    pickle_path: Path,
    training_output: Path,
    output_dir: Path,
    model_id: str,
    modality: str,
    selected_subtypes: int,
) -> SustainBundle:
    """Convert one selected full-data native result into the pickle-free inference format."""

    payload = load_numpy_result(pickle_path)
    required = {"samples_sequence", "samples_f"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"SuStaIn pickle is missing required fields: {missing}")
    sequence = np.asarray(payload["samples_sequence"])
    fractions = np.asarray(payload["samples_f"])
    if sequence.shape[0] != selected_subtypes or fractions.shape[0] != selected_subtypes:
        raise ValueError("selected_subtypes does not match the native result")
    return export_sustain_bundle(
        output_dir,
        model_id=model_id,
        modality=modality,
        selected_subtypes=selected_subtypes,
        preprocessor=load_training_preprocessor(training_output),
        samples_sequence=sequence,
        samples_f=fractions,
        source_pickle_sha256=sha256_file(pickle_path),
    )
