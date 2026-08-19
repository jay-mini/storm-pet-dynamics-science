from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from storm_pet.exceptions import ArtifactValidationError
from storm_pet.sustain.preprocessing import SustainPreprocessor

SCHEMA_VERSION = 1
DEPLOY_ARTIFACT_NAMES = {
    "metadata.json",
    "preprocessing.npz",
    "event_model.npz",
    "posterior_samples.npz",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class SustainBundle:
    root: Path
    model_id: str
    modality: str
    selected_subtypes: int
    preprocessor: SustainPreprocessor
    samples_sequence: np.ndarray
    samples_f: np.ndarray
    source_pickle_sha256: str | None = None

    def validate(self) -> None:
        sequence = np.asarray(self.samples_sequence)
        fractions = np.asarray(self.samples_f)
        if sequence.ndim != 3:
            raise ArtifactValidationError("samples_sequence must have shape subtype x event x sample")
        if fractions.ndim != 2:
            raise ArtifactValidationError("samples_f must have shape subtype x sample")
        if sequence.shape[0] != self.selected_subtypes:
            raise ArtifactValidationError("selected_subtypes disagrees with samples_sequence")
        if fractions.shape != (sequence.shape[0], sequence.shape[2]):
            raise ArtifactValidationError("samples_f shape disagrees with samples_sequence")
        if sequence.shape[1] != self.preprocessor.stage_max:
            raise ArtifactValidationError("posterior event count disagrees with z_vals")
        expected = np.arange(sequence.shape[1])
        for subtype in range(sequence.shape[0]):
            for sample in range(sequence.shape[2]):
                if not np.array_equal(np.sort(sequence[subtype, :, sample].astype(int)), expected):
                    raise ArtifactValidationError("each posterior sequence must be an event permutation")
        if np.any(~np.isfinite(fractions)) or np.any(fractions < 0):
            raise ArtifactValidationError("samples_f must be finite and non-negative")
        if not np.allclose(fractions.sum(axis=0), 1.0, atol=1e-6):
            raise ArtifactValidationError("subtype fractions must sum to one for each posterior sample")


def export_sustain_bundle(
    root: str | Path,
    *,
    model_id: str,
    modality: str,
    selected_subtypes: int,
    preprocessor: SustainPreprocessor,
    samples_sequence: np.ndarray,
    samples_f: np.ndarray,
    source_pickle_sha256: str | None = None,
) -> SustainBundle:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    bundle = SustainBundle(
        root=target,
        model_id=model_id,
        modality=modality,
        selected_subtypes=int(selected_subtypes),
        preprocessor=preprocessor,
        samples_sequence=np.asarray(samples_sequence),
        samples_f=np.asarray(samples_f),
        source_pickle_sha256=source_pickle_sha256,
    )
    bundle.validate()

    _write_json(
        target / "metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "model_id": model_id,
            "modality": modality,
            "selected_subtypes": selected_subtypes,
            "source_pickle_sha256": source_pickle_sha256,
            **preprocessor.metadata(),
        },
    )
    np.savez_compressed(
        target / "preprocessing.npz",
        feature_mask=preprocessor.feature_mask,
        control_mean=preprocessor.control_mean,
        control_std=preprocessor.control_std,
    )
    np.savez_compressed(target / "event_model.npz", z_vals=preprocessor.z_vals, z_max=preprocessor.z_max)
    np.savez_compressed(
        target / "posterior_samples.npz",
        samples_sequence=bundle.samples_sequence,
        samples_f=bundle.samples_f,
    )
    artifact_names = sorted(DEPLOY_ARTIFACT_NAMES)
    _write_json(
        target / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "model_id": model_id,
            "files": {name: {"sha256": _sha256(target / name)} for name in artifact_names},
        },
    )
    return bundle


def load_sustain_bundle(root: str | Path) -> SustainBundle:
    target = Path(root)
    required = sorted(DEPLOY_ARTIFACT_NAMES | {"manifest.json"})
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"incomplete SuStaIn bundle; missing: {', '.join(missing)}")
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported SuStaIn bundle schema version")
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != DEPLOY_ARTIFACT_NAMES:
        raise ArtifactValidationError("SuStaIn manifest does not list the exact deploy artifacts")
    if manifest.get("model_id") != metadata.get("model_id"):
        raise ArtifactValidationError("SuStaIn manifest and metadata model IDs disagree")
    for name, record in records.items():
        if name not in required or not (target / name).is_file():
            raise ArtifactValidationError(f"manifest references invalid file: {name}")
        if _sha256(target / name) != record.get("sha256"):
            raise ArtifactValidationError(f"checksum mismatch: {name}")

    # Object arrays are intentionally forbidden: deploy bundles must not execute pickle payloads.
    with np.load(target / "preprocessing.npz", allow_pickle=False) as prep, np.load(
        target / "event_model.npz", allow_pickle=False
    ) as event, np.load(target / "posterior_samples.npz", allow_pickle=False) as posterior:
        preprocessor = SustainPreprocessor(
            input_features=tuple(metadata["input_features"]),
            feature_mask=prep["feature_mask"].astype(bool),
            final_features=tuple(metadata["final_features"]),
            control_mean=prep["control_mean"],
            control_std=prep["control_std"],
            z_vals=event["z_vals"],
            z_max=event["z_max"],
            clip_min=float(metadata["clip_min"]),
            clip_max=float(metadata["clip_max"]),
            ddof=int(metadata["ddof"]),
        )
        bundle = SustainBundle(
            root=target,
            model_id=str(metadata["model_id"]),
            modality=str(metadata["modality"]),
            selected_subtypes=int(metadata["selected_subtypes"]),
            preprocessor=preprocessor,
            samples_sequence=posterior["samples_sequence"],
            samples_f=posterior["samples_f"],
            source_pickle_sha256=metadata.get("source_pickle_sha256"),
        )
    bundle.validate()
    return bundle


def inventory_incoming_pickles(root: str | Path) -> list[dict[str, Any]]:
    """Hash candidate pickle files without deserializing untrusted content."""

    incoming = Path(root)
    if not incoming.exists():
        return []
    records = []
    for path in sorted(p for p in incoming.rglob("*") if p.suffix.lower() in {".pickle", ".pkl"}):
        records.append(
            {
                "path": path.relative_to(incoming).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "deserialized": False,
            }
        )
    return records


def save_preprocessor_contract(
    root: str | Path,
    *,
    model_id: str,
    modality: str,
    preprocessor: SustainPreprocessor,
    recovery: dict[str, Any],
) -> Path:
    """Save the frozen training transform before the posterior pickle is available."""

    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "modality": modality,
        "status": "awaiting_pickle",
        "recovery": recovery,
        **preprocessor.metadata(),
    }
    _write_json(target / "metadata.json", metadata)
    np.savez_compressed(
        target / "preprocessing.npz",
        feature_mask=preprocessor.feature_mask,
        control_mean=preprocessor.control_mean,
        control_std=preprocessor.control_std,
    )
    np.savez_compressed(target / "event_model.npz", z_vals=preprocessor.z_vals, z_max=preprocessor.z_max)
    files = ["metadata.json", "preprocessing.npz", "event_model.npz"]
    _write_json(
        target / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "model_id": model_id,
            "status": "awaiting_pickle",
            "files": {name: {"sha256": _sha256(target / name)} for name in files},
        },
    )
    return target
