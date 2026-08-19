from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from storm_pet.exceptions import ArtifactValidationError

SCHEMA_VERSION = 1
DEPLOY_ARTIFACT_NAMES = {
    "metadata.json",
    "preprocessing.npz",
    "autoencoder_weights.npz",
    "cfm_weights.npz",
    "time_grid.npy",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_state(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    copied = {str(key): np.asarray(value).copy() for key, value in state.items()}
    if not copied or any(value.dtype.hasobject for value in copied.values()):
        raise ArtifactValidationError("OT-CFM deploy weights must be non-empty numeric arrays")
    if any(not np.isfinite(value).all() for value in copied.values()):
        raise ArtifactValidationError("OT-CFM deploy weights contain non-finite values")
    return copied


@dataclass(frozen=True)
class OTCFMBundle:
    root: Path
    model_id: str
    modality: str
    roi_columns: tuple[str, ...]
    roi_mean: np.ndarray
    roi_std: np.ndarray
    time_grid: np.ndarray
    stage_to_bin: dict[int, int]
    autoencoder_architecture: dict[str, int]
    cfm_architecture: dict[str, int]
    autoencoder_state: dict[str, np.ndarray]
    cfm_state: dict[str, np.ndarray]

    @property
    def final_bin(self) -> int:
        return int(round(float(self.time_grid[-1])))

    def validate(self) -> None:
        if len(self.roi_columns) != 68 or len(set(self.roi_columns)) != 68:
            raise ArtifactValidationError("OT-CFM deploy bundle must contain 68 unique ROI columns")
        mean = np.asarray(self.roi_mean, dtype=float).reshape(-1)
        std = np.asarray(self.roi_std, dtype=float).reshape(-1)
        if mean.shape != (68,) or std.shape != (68,):
            raise ArtifactValidationError("OT-CFM scaler must match the 68 ROI contract")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ArtifactValidationError("OT-CFM scaler is invalid")
        grid = np.asarray(self.time_grid, dtype=float).reshape(-1)
        if len(grid) < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
            raise ArtifactValidationError("OT-CFM time grid must be finite and strictly increasing")
        if not np.isclose(grid[0], 0.0) or not np.isclose(grid[-1], self.final_bin):
            raise ArtifactValidationError("OT-CFM time grid must span integer dynamics bins")
        stages = sorted(self.stage_to_bin)
        if not stages or stages != list(range(stages[-1] + 1)):
            raise ArtifactValidationError("OT-CFM stage-to-bin map must cover every stage from zero")
        bins = [self.stage_to_bin[stage] for stage in stages]
        if bins != sorted(bins) or min(bins) != 0 or max(bins) != self.final_bin:
            raise ArtifactValidationError("OT-CFM stage-to-bin map is inconsistent with time grid")
        if int(self.autoencoder_architecture.get("input_dim", -1)) != 68:
            raise ArtifactValidationError("OT-CFM autoencoder input dimension must be 68")
        if int(self.autoencoder_architecture.get("latent_dim", -1)) != int(
            self.cfm_architecture.get("latent_dim", -2)
        ):
            raise ArtifactValidationError("autoencoder and CFM latent dimensions disagree")
        _copy_state(self.autoencoder_state)
        _copy_state(self.cfm_state)


def export_ot_cfm_bundle(
    root: str | Path,
    *,
    model_id: str,
    modality: str,
    roi_columns: list[str] | tuple[str, ...],
    roi_mean: np.ndarray,
    roi_std: np.ndarray,
    time_grid: np.ndarray,
    stage_to_bin: Mapping[int, int],
    autoencoder_architecture: Mapping[str, int],
    cfm_architecture: Mapping[str, int],
    autoencoder_state: Mapping[str, np.ndarray],
    cfm_state: Mapping[str, np.ndarray],
    source: Mapping[str, Any],
) -> OTCFMBundle:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    bundle = OTCFMBundle(
        root=target,
        model_id=model_id,
        modality=modality,
        roi_columns=tuple(str(column) for column in roi_columns),
        roi_mean=np.asarray(roi_mean, dtype=np.float32).reshape(-1),
        roi_std=np.asarray(roi_std, dtype=np.float32).reshape(-1),
        time_grid=np.asarray(time_grid, dtype=np.float32).reshape(-1),
        stage_to_bin={int(stage): int(bin_index) for stage, bin_index in stage_to_bin.items()},
        autoencoder_architecture={str(key): int(value) for key, value in autoencoder_architecture.items()},
        cfm_architecture={str(key): int(value) for key, value in cfm_architecture.items()},
        autoencoder_state=_copy_state(autoencoder_state),
        cfm_state=_copy_state(cfm_state),
    )
    bundle.validate()
    _write_json(
        target / "metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "model_id": model_id,
            "modality": modality,
            "roi_columns": list(bundle.roi_columns),
            "stage_to_bin": {str(key): value for key, value in bundle.stage_to_bin.items()},
            "autoencoder_architecture": bundle.autoencoder_architecture,
            "cfm_architecture": bundle.cfm_architecture,
            "source": dict(source),
        },
    )
    np.savez_compressed(
        target / "preprocessing.npz",
        roi_mean=bundle.roi_mean,
        roi_std=bundle.roi_std,
    )
    np.savez_compressed(target / "autoencoder_weights.npz", **bundle.autoencoder_state)
    np.savez_compressed(target / "cfm_weights.npz", **bundle.cfm_state)
    np.save(target / "time_grid.npy", bundle.time_grid, allow_pickle=False)
    _write_json(
        target / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "model_id": model_id,
            "files": {
                name: {
                    "sha256": sha256_file(target / name),
                    "size_bytes": (target / name).stat().st_size,
                }
                for name in sorted(DEPLOY_ARTIFACT_NAMES)
            },
        },
    )
    return bundle


def load_ot_cfm_bundle(root: str | Path) -> OTCFMBundle:
    target = Path(root)
    required = DEPLOY_ARTIFACT_NAMES | {"manifest.json"}
    missing = sorted(name for name in required if not (target / name).is_file())
    if missing:
        raise ArtifactValidationError(f"incomplete OT-CFM deploy bundle; missing: {', '.join(missing)}")
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported OT-CFM deploy bundle schema version")
    if metadata.get("model_id") != manifest.get("model_id"):
        raise ArtifactValidationError("OT-CFM manifest and metadata model IDs disagree")
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != DEPLOY_ARTIFACT_NAMES:
        raise ArtifactValidationError("OT-CFM manifest does not list the exact deploy artifacts")
    for name, record in records.items():
        path = target / name
        if sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get("size_bytes"):
            raise ArtifactValidationError(f"checksum mismatch: {name}")
    with np.load(target / "preprocessing.npz", allow_pickle=False) as preprocessing, np.load(
        target / "autoencoder_weights.npz", allow_pickle=False
    ) as autoencoder, np.load(target / "cfm_weights.npz", allow_pickle=False) as cfm:
        bundle = OTCFMBundle(
            root=target,
            model_id=str(metadata["model_id"]),
            modality=str(metadata["modality"]),
            roi_columns=tuple(str(column) for column in metadata["roi_columns"]),
            roi_mean=preprocessing["roi_mean"].copy(),
            roi_std=preprocessing["roi_std"].copy(),
            time_grid=np.load(target / "time_grid.npy", allow_pickle=False),
            stage_to_bin={int(key): int(value) for key, value in metadata["stage_to_bin"].items()},
            autoencoder_architecture={
                str(key): int(value) for key, value in metadata["autoencoder_architecture"].items()
            },
            cfm_architecture={
                str(key): int(value) for key, value in metadata["cfm_architecture"].items()
            },
            autoencoder_state={key: autoencoder[key].copy() for key in autoencoder.files},
            cfm_state={key: cfm[key].copy() for key in cfm.files},
        )
    bundle.validate()
    return bundle
