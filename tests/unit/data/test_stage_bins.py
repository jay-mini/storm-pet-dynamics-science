import pandas as pd

from storm_pet.data.stage_bins import assign_stage_bins, contiguous_equal_count_bins


def test_greedy_bins_are_contiguous_and_cover_all_positive_stages() -> None:
    bins = contiguous_equal_count_bins(
        pd.Series([5, 5, 10, 10], index=[1, 2, 3, 4]),
        desired_bins=2,
        minimum_samples_per_bin=1,
    )
    assert [(item.stage_start, item.stage_end) for item in bins] == [(1, 3), (4, 4)]
    assert sum(item.sample_count for item in bins) == 30


def test_stage_zero_becomes_shared_root_and_positive_bins_shift_by_one() -> None:
    frame = pd.DataFrame(
        {
            "sustain_stage": [0, 1, 2, 3, 4],
            "sustain_subtype": [0, 0, 1, 0, 1],
        }
    )
    output, definitions = assign_stage_bins(
        frame, desired_positive_bins=2, minimum_samples_per_bin=1
    )
    assert output["dynamics_bin"].tolist() == [0, 1, 1, 2, 2]
    assert output["use_in_otfm"].tolist() == [False, True, True, True, True]
    assert definitions[["stage_start", "stage_end"]].values.tolist() == [[1, 2], [3, 4]]


def test_fixed_main_text_ranges_are_used_without_refitting() -> None:
    frame = pd.DataFrame(
        {"sustain_stage": [0, 1, 3, 6], "sustain_subtype": [0, 0, 1, 1]}
    )
    output, definitions = assign_stage_bins(
        frame, fixed_stage_ranges=((1, 2), (3, 6))
    )
    assert output["dynamics_bin"].tolist() == [0, 1, 2, 2]
    assert definitions[["stage_start", "stage_end"]].values.tolist() == [[1, 2], [3, 6]]
