from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StageBin:
    index: int
    stage_start: int
    stage_end: int
    sample_count: int


def contiguous_equal_count_bins(
    stage_count: pd.Series,
    *,
    desired_bins: int,
    minimum_samples_per_bin: int,
) -> tuple[StageBin, ...]:
    """Partition ordered positive stages using the paper's deterministic greedy rule."""

    if desired_bins < 1 or minimum_samples_per_bin < 1:
        raise ValueError("bin counts and minimum samples must be positive")
    counts = pd.to_numeric(stage_count, errors="coerce").dropna().astype(int).sort_index()
    if counts.empty:
        return ()
    if any(int(stage) <= 0 for stage in counts.index) or any(count < 0 for count in counts):
        raise ValueError("stage_count must contain non-negative counts for positive stages")
    total = int(counts.sum())
    n_bins = max(
        1,
        min(desired_bins, max(1, total // minimum_samples_per_bin), len(counts)),
    )
    stages = [int(stage) for stage in counts.index]
    values = [int(count) for count in counts]
    ranges: list[tuple[int, int, int]] = []
    start = stages[0]
    accumulated = 0
    remaining_total = total
    remaining_bins = n_bins
    for position, (stage, count) in enumerate(zip(stages, values, strict=True)):
        accumulated += count
        stages_left = len(stages) - position - 1
        should_cut = (
            remaining_bins > 1
            and accumulated >= remaining_total / remaining_bins
            and stages_left >= remaining_bins - 1
        )
        if should_cut:
            ranges.append((start, stage, accumulated))
            remaining_total -= accumulated
            remaining_bins -= 1
            start = stages[position + 1]
            accumulated = 0
    ranges.append((start, stages[-1], accumulated))
    return tuple(StageBin(index, *values) for index, values in enumerate(ranges))


def assign_stage_bins(
    frame: pd.DataFrame,
    *,
    stage_column: str = "sustain_stage",
    subtype_column: str = "sustain_subtype",
    desired_positive_bins: int = 4,
    minimum_samples_per_bin: int = 25,
    fixed_stage_ranges: tuple[tuple[int, int], ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add shared positive-stage bins; stage 0 remains the common OT-CFM root."""

    missing = {stage_column, subtype_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"stage-binning input is missing columns: {sorted(missing)}")
    output = frame.copy()
    stage = pd.to_numeric(output[stage_column], errors="coerce")
    subtype = pd.to_numeric(output[subtype_column], errors="coerce")
    valid = stage.notna() & subtype.notna()
    positive = valid & (stage > 0)
    positive_counts = stage.loc[positive].astype(int).value_counts().sort_index()
    if fixed_stage_ranges is None:
        bins = contiguous_equal_count_bins(
            positive_counts,
            desired_bins=desired_positive_bins,
            minimum_samples_per_bin=minimum_samples_per_bin,
        )
    else:
        if not fixed_stage_ranges or any(start < 1 or end < start for start, end in fixed_stage_ranges):
            raise ValueError("fixed stage ranges must be non-empty increasing positive intervals")
        covered = [stage_value for start, end in fixed_stage_ranges for stage_value in range(start, end + 1)]
        if covered != list(range(covered[-1] + 1))[1:]:
            raise ValueError("fixed stage ranges must contiguously cover stages 1 through the maximum")
        observed = set(positive_counts.index.astype(int))
        if observed and (min(observed) < 1 or max(observed) > covered[-1]):
            raise ValueError("observed positive stage falls outside the fixed paper ranges")
        bins = tuple(
            StageBin(
                index=index,
                stage_start=start,
                stage_end=end,
                sample_count=int(positive_counts.reindex(range(start, end + 1), fill_value=0).sum()),
            )
            for index, (start, end) in enumerate(fixed_stage_ranges)
        )
    output["coarse_stage_bin"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["coarse_stage_label"] = pd.Series(pd.NA, index=output.index, dtype="string")
    for item in bins:
        selected = positive & stage.between(item.stage_start, item.stage_end, inclusive="both")
        output.loc[selected, "coarse_stage_bin"] = item.index
        output.loc[selected, "coarse_stage_label"] = (
            f"bin{item.index}: stage{item.stage_start}-{item.stage_end}"
        )
    output["use_in_otfm"] = output["coarse_stage_bin"].notna()
    output["dynamics_bin"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output.loc[valid & (stage == 0), "dynamics_bin"] = 0
    output.loc[output["coarse_stage_bin"].notna(), "dynamics_bin"] = (
        output.loc[output["coarse_stage_bin"].notna(), "coarse_stage_bin"] + 1
    )
    definitions = pd.DataFrame(
        [
            {
                "coarse_stage_bin": item.index,
                "dynamics_bin": item.index + 1,
                "stage_start": item.stage_start,
                "stage_end": item.stage_end,
                "n_samples_total": item.sample_count,
                "label": f"stage{item.stage_start}-{item.stage_end}",
            }
            for item in bins
        ]
    )
    return output, definitions
