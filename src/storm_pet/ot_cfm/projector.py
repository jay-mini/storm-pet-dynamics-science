from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from storm_pet.exceptions import ArtifactValidationError, ConfigurationError
from storm_pet.ot_cfm.artifacts import OTCFMBundle, load_ot_cfm_bundle


@dataclass(frozen=True)
class OTCFMBinSnapshot:
    dynamics_time: float
    roi_values: tuple[float, ...]


@dataclass(frozen=True)
class OTCFMProjection:
    scan_id: str
    subtype: int
    stage: int
    current_bin: int
    final_bin: int
    used_ot_cfm: bool
    roi_columns: tuple[str, ...]
    observed_roi: tuple[float, ...]
    final_roi: tuple[float, ...]
    snapshots: tuple[OTCFMBinSnapshot, ...]


class OTCFMProjector:
    """Project one observed 68-ROI scan to the checkpoint's final dynamics bin."""

    OUTPUT_STEP = 0.1

    def __init__(self, bundle: OTCFMBundle) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - installation error
            raise ConfigurationError("OT-CFM web inference requires the model dependencies") from error

        from storm_pet.ot_cfm.models import AutoEncoder, ConditionalVectorField

        bundle.validate()
        self.bundle = bundle
        self.torch = torch
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        ae = bundle.autoencoder_architecture
        self.autoencoder = AutoEncoder(
            input_dim=ae["input_dim"],
            latent_dim=ae["latent_dim"],
            hidden_width=ae["hidden_width"],
            n_subtypes=ae["n_subtypes"],
            n_bins=ae["n_bins"],
            dropout=0.0,
        ).to(self.device)
        cfm = bundle.cfm_architecture
        self.cfm = ConditionalVectorField(
            dim=cfm["latent_dim"],
            n_subtypes=cfm["n_subtypes"],
            hidden_width=cfm["hidden_width"],
        ).to(self.device)
        self.autoencoder.load_state_dict(self._torch_state(bundle.autoencoder_state))
        self.cfm.load_state_dict(self._torch_state(bundle.cfm_state))
        self.autoencoder.eval()
        self.cfm.eval()

    @classmethod
    def from_bundle(cls, root: str | Path) -> "OTCFMProjector":
        return cls(load_ot_cfm_bundle(root))

    def _torch_state(self, state: Mapping[str, np.ndarray]):
        return {key: self.torch.from_numpy(np.asarray(value).copy()) for key, value in state.items()}

    def schema(self) -> dict[str, object]:
        return {
            "ot_cfm_required_columns": list(self.bundle.roi_columns),
            "stage_to_bin": {str(key): value for key, value in self.bundle.stage_to_bin.items()},
            "final_bin": self.bundle.final_bin,
        }

    def project(
        self,
        scan_id: str,
        values: Mapping[str, float],
        *,
        subtype: int,
        stage: int,
    ) -> OTCFMProjection:
        missing = [column for column in self.bundle.roi_columns if column not in values]
        if missing:
            raise ConfigurationError(
                "missing OT-CFM ROI columns: " + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            )
        if stage not in self.bundle.stage_to_bin:
            raise ConfigurationError(f"stage {stage} is outside the OT-CFM training contract")
        if subtype < 0 or subtype >= self.bundle.cfm_architecture["n_subtypes"]:
            raise ConfigurationError("subtype is outside the OT-CFM training contract")
        observed = np.asarray([values[column] for column in self.bundle.roi_columns], dtype=np.float32)
        if observed.shape != (68,) or not np.isfinite(observed).all():
            raise ConfigurationError("OT-CFM ROI values must be finite numbers")
        current_bin = self.bundle.stage_to_bin[stage]
        final_bin = self.bundle.final_bin
        observed_tuple = tuple(float(value) for value in observed)
        if current_bin == final_bin:
            snapshot = OTCFMBinSnapshot(float(current_bin), observed_tuple)
            return OTCFMProjection(
                scan_id=scan_id,
                subtype=subtype,
                stage=stage,
                current_bin=current_bin,
                final_bin=final_bin,
                used_ot_cfm=False,
                roi_columns=self.bundle.roi_columns,
                observed_roi=observed_tuple,
                final_roi=observed_tuple,
                snapshots=(snapshot,),
            )

        torch = self.torch
        standardized = (observed - self.bundle.roi_mean) / self.bundle.roi_std
        with torch.inference_mode():
            x = torch.from_numpy(standardized[None, :]).to(self.device)
            z = self.autoencoder.encode(x)
            grid = self._projection_grid(current_bin, final_bin)
            condition = torch.zeros((1, self.bundle.cfm_architecture["n_subtypes"]), device=self.device)
            condition[0, subtype] = 1.0
            latent_states = [z]
            for left, right in zip(grid[:-1], grid[1:]):
                time_value = torch.full((1,), float(left), device=self.device)
                velocity = self.cfm(z, time_value, condition)
                z = z + float(right - left) * velocity
                latent_states.append(z)
            output_times = self._output_times(current_bin, final_bin)[1:]
            selected = [
                latent_states[int(np.flatnonzero(np.isclose(grid, output_time))[0])]
                for output_time in output_times
            ]
            decoded_standardized = self.autoencoder.decode(torch.cat(selected, dim=0)).cpu().numpy()
        decoded = decoded_standardized * self.bundle.roi_std[None, :] + self.bundle.roi_mean[None, :]
        if not np.isfinite(decoded).all():
            raise ArtifactValidationError("OT-CFM produced non-finite decoded ROI values")
        snapshots = [OTCFMBinSnapshot(float(current_bin), observed_tuple)]
        snapshots.extend(
            OTCFMBinSnapshot(
                float(output_time),
                tuple(float(value) for value in decoded[index]),
            )
            for index, output_time in enumerate(output_times)
        )
        return OTCFMProjection(
            scan_id=scan_id,
            subtype=subtype,
            stage=stage,
            current_bin=current_bin,
            final_bin=final_bin,
            used_ot_cfm=True,
            roi_columns=self.bundle.roi_columns,
            observed_roi=observed_tuple,
            final_roi=snapshots[-1].roi_values,
            snapshots=tuple(snapshots),
        )

    def _projection_grid(self, current_bin: int, final_bin: int) -> np.ndarray:
        future = self.bundle.time_grid[
            (self.bundle.time_grid > current_bin) & (self.bundle.time_grid < final_bin)
        ]
        return np.unique(np.concatenate((self._output_times(current_bin, final_bin), future)))

    def _output_times(self, current_bin: int, final_bin: int) -> np.ndarray:
        step_count = int(round((final_bin - current_bin) / self.OUTPUT_STEP))
        return np.linspace(current_bin, final_bin, step_count + 1, dtype=np.float32)
