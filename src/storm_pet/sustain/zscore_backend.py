from __future__ import annotations

import numpy as np


class ZScoreSustainInferenceBackend:
    """Inference-only implementation of the pySuStaIn z-score likelihood.

    The equations and posterior-sample selection follow ucl-pond/pySuStaIn's
    MIT-licensed ``ZscoreSustain`` and ``AbstractSustain`` implementations.
    Training, plotting, multiprocessing, and pickle handling are deliberately
    excluded from this deployment backend.
    """

    def __init__(self, z_vals: np.ndarray, z_max: np.ndarray):
        values = np.asarray(z_vals, dtype=float)
        maxima = np.asarray(z_max, dtype=float).reshape(-1)
        if values.ndim != 2 or maxima.shape != (values.shape[0],):
            raise ValueError("z_vals and z_max have incompatible shapes")
        stage_zscore = values.T.flatten()
        selected = stage_zscore > 0
        self.stage_zscore = stage_zscore[selected]
        indices = np.tile(np.arange(values.shape[0])[:, None], (1, values.shape[1])).T.flatten()
        self.stage_biomarker_index = indices[selected]
        self.z_max = maxima
        self.n_biomarkers = values.shape[0]
        self.n_stages = self.stage_zscore.size

    @classmethod
    def from_bundle(cls, bundle):
        return cls(bundle.preprocessor.z_vals, bundle.preprocessor.z_max)

    @staticmethod
    def _linspace(a: float, b: float, count: int) -> np.ndarray:
        if count == 1:
            return np.asarray([a], dtype=float)
        return a + (b - a) / (count - 1.0) * np.arange(count)

    def _stage_values(self, sequence: np.ndarray) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=int).reshape(-1)
        if sequence.shape != (self.n_stages,):
            raise ValueError("sequence event count does not match the z-score model")
        inverse = np.empty(self.n_stages, dtype=int)
        inverse[sequence] = np.arange(self.n_stages)
        point_value = np.zeros((self.n_biomarkers, self.n_stages + 2))
        stage_axis = np.arange(self.n_stages + 2)

        for biomarker in range(self.n_biomarkers):
            positions = np.concatenate(
                ([0], inverse[self.stage_biomarker_index == biomarker], [self.n_stages])
            )
            values = np.concatenate(
                ([0.0], self.stage_zscore[self.stage_biomarker_index == biomarker], [self.z_max[biomarker]])
            )
            for interval in range(len(positions) - 1):
                if interval == 0:
                    target = stage_axis[positions[interval] : positions[interval + 1] + 2]
                    count = positions[interval + 1] - positions[interval] + 2
                else:
                    target = stage_axis[positions[interval] + 1 : positions[interval + 1] + 2]
                    count = positions[interval + 1] - positions[interval] + 1
                point_value[biomarker, target] = self._linspace(
                    values[interval], values[interval + 1], count
                )

        return 0.5 * (point_value[:, :-1] + point_value[:, 1:])

    def _likelihood_stage(self, data: np.ndarray, sequence: np.ndarray) -> np.ndarray:
        stage_value = self._stage_values(sequence)
        residual = data[:, :, None] - stage_value[None, :, :]
        log_normalizer = np.log(1.0 / np.sqrt(2.0 * np.pi))
        log_stage_prior = np.log(1.0 / (self.n_stages + 1.0))
        return np.exp(log_stage_prior + np.sum(log_normalizer - 0.5 * residual**2, axis=1))

    def _likelihood(
        self, data: np.ndarray, sequences: np.ndarray, fractions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_subtypes = sequences.shape[0]
        stage_subtype = np.zeros((data.shape[0], self.n_stages + 1, n_subtypes))
        for subtype in range(n_subtypes):
            stage_subtype[:, :, subtype] = self._likelihood_stage(data, sequences[subtype])
        weighted = stage_subtype * np.asarray(fractions)[None, None, :]
        return np.sum(weighted, axis=2), np.sum(weighted, axis=1), stage_subtype

    def subtype_and_stage_individuals_newData(
        self,
        data_new: np.ndarray,
        samples_sequence: np.ndarray,
        samples_f: np.ndarray,
        n_samples: int,
    ) -> tuple[np.ndarray, ...]:
        data = np.asarray(data_new, dtype=float)
        sequence_samples = np.asarray(samples_sequence)
        fraction_samples = np.asarray(samples_f)
        if data.ndim != 2 or data.shape[1] != self.n_biomarkers:
            raise ValueError("new data do not match the z-score model features")
        if n_samples < 1 or n_samples > sequence_samples.shape[2]:
            raise ValueError("invalid posterior sample count")

        selected = np.round(np.linspace(0, sequence_samples.shape[2] - 1, n_samples)).astype(int)
        subtype_order = np.argsort(np.mean(fraction_samples, axis=1))[::-1]
        n_subjects = data.shape[0]
        n_subtypes = sequence_samples.shape[0]
        prob_subtype_stage = np.zeros((n_subjects, self.n_stages + 1, n_subtypes))
        prob_subtype = np.zeros((n_subjects, n_subtypes))
        prob_stage = np.zeros((n_subjects, self.n_stages + 1))

        for iteration, sample in enumerate(selected):
            total_stage, total_subtype, stage_subtype = self._likelihood(
                data,
                sequence_samples[subtype_order, :, sample],
                fraction_samples[subtype_order, sample],
            )
            subtype_norm = total_subtype / np.sum(total_subtype, axis=1, keepdims=True)
            stage_norm = total_stage / np.sum(total_stage, axis=1, keepdims=True)
            joint_norm = stage_subtype / np.sum(stage_subtype, axis=(1, 2), keepdims=True)
            weight_old = iteration / (iteration + 1.0)
            weight_new = 1.0 / (iteration + 1.0)
            prob_subtype_stage = weight_old * prob_subtype_stage + weight_new * joint_norm
            prob_subtype = weight_old * prob_subtype + weight_new * subtype_norm
            prob_stage = weight_old * prob_stage + weight_new * stage_norm

        ml_subtype = np.argmax(prob_subtype, axis=1).astype(float).reshape(-1, 1)
        prob_ml_subtype = np.take_along_axis(prob_subtype, ml_subtype.astype(int), axis=1)
        selected_joint = np.take_along_axis(
            prob_subtype_stage, ml_subtype.astype(int)[:, None, :], axis=2
        ).squeeze(2)
        ml_stage = np.argmax(selected_joint, axis=1).astype(float).reshape(-1, 1)
        prob_ml_stage = np.take_along_axis(selected_joint, ml_stage.astype(int), axis=1)
        return (
            ml_subtype,
            prob_ml_subtype,
            ml_stage,
            prob_ml_stage,
            prob_subtype,
            prob_stage,
            prob_subtype_stage,
        )


class PreparedZScoreSustainBackend:
    """Precompute posterior trajectories once for low-latency web inference."""

    def __init__(
        self,
        z_vals: np.ndarray,
        z_max: np.ndarray,
        samples_sequence: np.ndarray,
        samples_f: np.ndarray,
        *,
        n_samples: int = 1000,
        posterior_chunk_size: int = 50,
    ) -> None:
        base = ZScoreSustainInferenceBackend(z_vals, z_max)
        sequences = np.asarray(samples_sequence)
        fractions = np.asarray(samples_f, dtype=float)
        if sequences.ndim != 3 or fractions.shape != (
            sequences.shape[0],
            sequences.shape[2],
        ):
            raise ValueError("posterior arrays have incompatible shapes")
        if n_samples < 1 or n_samples > sequences.shape[2]:
            raise ValueError("invalid posterior sample count")
        if posterior_chunk_size < 1:
            raise ValueError("posterior_chunk_size must be positive")

        selected = np.round(np.linspace(0, sequences.shape[2] - 1, n_samples)).astype(int)
        subtype_order = np.argsort(np.mean(fractions, axis=1))[::-1]
        ordered_sequences = sequences[subtype_order][:, :, selected].transpose(2, 0, 1)
        self.fractions = fractions[subtype_order][:, selected].T
        self.stage_values = np.empty(
            (
                n_samples,
                sequences.shape[0],
                base.n_biomarkers,
                base.n_stages + 1,
            ),
            dtype=float,
        )
        for sample in range(n_samples):
            for subtype in range(sequences.shape[0]):
                self.stage_values[sample, subtype] = base._stage_values(
                    ordered_sequences[sample, subtype]
                )

        self.n_biomarkers = base.n_biomarkers
        self.n_stages = base.n_stages
        self.n_subtypes = sequences.shape[0]
        self.n_samples = n_samples
        self.posterior_chunk_size = posterior_chunk_size

    @classmethod
    def from_bundle(
        cls,
        bundle,
        *,
        n_samples: int = 1000,
        posterior_chunk_size: int = 50,
    ):
        return cls(
            bundle.preprocessor.z_vals,
            bundle.preprocessor.z_max,
            bundle.samples_sequence,
            bundle.samples_f,
            n_samples=min(n_samples, bundle.samples_sequence.shape[2]),
            posterior_chunk_size=posterior_chunk_size,
        )

    def subtype_and_stage_individuals_newData(self, data_new: np.ndarray) -> tuple[np.ndarray, ...]:
        data = np.asarray(data_new, dtype=float)
        if data.ndim != 2 or data.shape[1] != self.n_biomarkers:
            raise ValueError("new data do not match the prepared z-score model")
        if not np.isfinite(data).all():
            raise ValueError("new data contain NaN or infinity")

        n_subjects = data.shape[0]
        subtype_sum = np.zeros((n_subjects, self.n_subtypes), dtype=float)
        stage_sum = np.zeros((n_subjects, self.n_stages + 1), dtype=float)
        joint_sum = np.zeros(
            (n_subjects, self.n_stages + 1, self.n_subtypes), dtype=float
        )
        log_normalizer = np.log(1.0 / np.sqrt(2.0 * np.pi))
        log_stage_prior = np.log(1.0 / (self.n_stages + 1.0))

        for start in range(0, self.n_samples, self.posterior_chunk_size):
            stop = min(start + self.posterior_chunk_size, self.n_samples)
            stage_values = self.stage_values[start:stop]
            residual = (
                data[:, None, None, :, None] - stage_values[None, :, :, :, :]
            )
            likelihood = np.exp(
                log_stage_prior
                + np.sum(log_normalizer - 0.5 * residual**2, axis=3)
            )
            weighted = likelihood * self.fractions[None, start:stop, :, None]

            total_subtype = weighted.sum(axis=3)
            subtype_denominator = total_subtype.sum(axis=2, keepdims=True)
            total_stage = weighted.sum(axis=2)
            stage_denominator = total_stage.sum(axis=2, keepdims=True)
            joint_denominator = likelihood.sum(axis=(2, 3), keepdims=True)
            if (
                np.any(subtype_denominator <= 0)
                or np.any(stage_denominator <= 0)
                or np.any(joint_denominator <= 0)
            ):
                raise FloatingPointError("SuStaIn likelihood normalization underflowed")

            subtype_sum += (total_subtype / subtype_denominator).sum(axis=1)
            stage_sum += (total_stage / stage_denominator).sum(axis=1)
            joint_sum += (
                likelihood / joint_denominator
            ).sum(axis=1).transpose(0, 2, 1)

        prob_subtype = subtype_sum / self.n_samples
        prob_stage = stage_sum / self.n_samples
        prob_subtype_stage = joint_sum / self.n_samples
        ml_subtype = np.argmax(prob_subtype, axis=1).astype(float).reshape(-1, 1)
        prob_ml_subtype = np.take_along_axis(
            prob_subtype, ml_subtype.astype(int), axis=1
        )
        selected_joint = np.take_along_axis(
            prob_subtype_stage,
            ml_subtype.astype(int)[:, None, :],
            axis=2,
        ).squeeze(2)
        ml_stage = np.argmax(selected_joint, axis=1).astype(float).reshape(-1, 1)
        prob_ml_stage = np.take_along_axis(
            selected_joint, ml_stage.astype(int), axis=1
        )
        return (
            ml_subtype,
            prob_ml_subtype,
            ml_stage,
            prob_ml_stage,
            prob_subtype,
            prob_stage,
            prob_subtype_stage,
        )
