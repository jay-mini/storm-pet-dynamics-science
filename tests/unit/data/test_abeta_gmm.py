import numpy as np
import pandas as pd

from storm_pet.data.abeta_gmm import fit_abeta_gmm


def synthetic_summary_values() -> pd.DataFrame:
    generator = np.random.default_rng(42)
    negative = generator.normal(0.95, 0.04, 200)
    positive = generator.normal(1.45, 0.07, 180)
    return pd.DataFrame({"SUMMARY_SUVR": np.concatenate([negative, positive, [np.nan]])})


def test_gmm_high_mean_component_is_positive_and_missing_is_preserved() -> None:
    fit = fit_abeta_gmm(synthetic_summary_values())
    assert fit.means[fit.positive_component] > fit.means[fit.negative_component]
    assert fit.labels.iloc[-1] is pd.NA
    assert fit.labels.iloc[:200].mean() == 0
    assert fit.labels.iloc[200:-1].mean() == 1


def test_gmm_is_deterministic_and_cutoff_is_between_component_means() -> None:
    data = synthetic_summary_values()
    first = fit_abeta_gmm(data)
    second = fit_abeta_gmm(data)
    assert first.means == second.means
    assert first.labels.equals(second.labels)
    assert min(first.means) < first.posterior_cutoff < max(first.means)

